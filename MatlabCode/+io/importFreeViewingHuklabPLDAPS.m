function D = importFreeViewingHuklabPLDAPS(S)
% this function imports the data required for the FREEVIEWING project
% It shares the same fundamentals as
disp('FREE VIEWING IMPORT FOR HUKLAB DATASHARE')
disp('THIS MUST BE RUN ON A COMPUTER CONNECTED TO THE HUKLAB DATASHARE DRIVE FOLDER')
disp('REQUIRES MARMOPIPE CODE IN THE PATH')

ip = struct();
ip.Results.wheelsmoothing = 15; % HARD CODED SMOOTHING FOR SPEED

% get paths
SERVER_DATA_DIR = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');

if contains(S.rawFilePath, SERVER_DATA_DIR)
    DataFolder = S.rawFilePath;
else
    DataFolder = fullfile(SERVER_DATA_DIR, S.rawFilePath);
end

assert(exist(DataFolder, 'dir')==7, 'importFreeViewing: raw data path does not exist')

pdsfiles = dir(fullfile(DataFolder, '*.PDS'));
kwefiles = dir(fullfile(DataFolder, '*.kwe'));



D = struct();
D.treadTime = [];
D.treadSpeed = [];
D.treadPos = [];
D.GratingOnsets = [];
D.GratingOffsets = [];
D.GratingContrast = [];
D.GratingDirections = [];
D.GratingFrequency = [];
D.GratingSpeeds = [];
D.frameTimes = [];
D.framePhase = [];
D.frameContrast = [];
D.eyeTime = [];
D.eyePos = [];
D.eyeLabels = [];


for iFile = 1:numel(pdsfiles)
    
    PDS = load(fullfile(pdsfiles(iFile).folder, pdsfiles(iFile).name), '-mat');
    if isfield(PDS, 'tmpFile')
        PDS = PDS.tmpFile;
    end
    
    if numel(kwefiles) > 1
        [~, id] = min( ([kwefiles.datenum] - pdsfiles(iFile).datenum).^2);
        filenameE = fullfile(DataFolder, kwefiles(id).name);
    else
        filenameE = fullfile(DataFolder,kwefiles(1).name);
    end
    
    % sync clock
    timestamps = h5read(filenameE, '/event_types/TTL/events/time_samples');
    highlow = h5read(filenameE, '/event_types/TTL/events/user_data/eventID');
    bitNumber = h5read(filenameE, '/event_types/TTL/events/user_data/event_channels');
    
    strobeSet=find(bitNumber==7 & highlow==1);
    strobeUnset=find(bitNumber==7 & highlow==0);
    strobeUnset=[1; strobeUnset];
    
    value=nan(size(strobeSet));
    for iStrobe=1:length(strobeSet)
        ts=timestamps <= timestamps(strobeSet(iStrobe)) & timestamps >= timestamps(strobeUnset(iStrobe)) & bitNumber~=7;
        value(iStrobe)=sum(2.^bitNumber(ts) .* highlow(ts));
    end
    
    eventTimes=double(timestamps(strobeSet));
    eventValues = value;
    flagBits = nan(size(value));
    flagData = value;
    invertedBits = false;
    switchbits = false;
    
    %eventSamples depends on recording number.
    location='/recordings';
    info=h5info(filenameE,location);
    nRecs=length(info.Groups);
    for iRec=1:nRecs
        iRecNumber=(info.Groups(iRec).Name(13:end));
        if str2double(iRecNumber)==0
            st_index=strcmp('start_time',{info.Groups(iRec).Attributes.Name});
            recStartTime = double(info.Groups(iRec).Attributes(st_index).Value);
            eventSamples=eventTimes-recStartTime;
            sample_rate = h5readatt(filenameE, sprintf('%s/%s', location, iRecNumber), 'sample_rate');
        end
    end
    
    ephysData = struct('eventTimes', eventTimes, 'eventSamples', eventSamples, ...
        'eventValues', eventValues, 'flagBits', flagBits, 'flagData', flagData, ...
        'invertedBits', invertedBits, 'switchbits', switchbits, 'recStartTime', recStartTime);
    
    [EP2PTBfit, EP2PTB, PTB2EP, maxreconstructionerror] = ephys.syncGeneralClock(PDS, ephysData);
    
    if isempty(maxreconstructionerror)
        continue % skip file
    end
    
    % loop over trials
    numTrials = numel(PDS.data);
    
    for iTrial = 1:numTrials
        fprintf('Trial %d/%d\n', iTrial, numTrials)
        fnames = fieldnames(PDS.data{iTrial});
        gfields = fnames(contains(fnames, 'grate'));
        %     assert(numel(gfields) == numel(PDS.data{iTrial}.pmBase.condsShown), "number of gratings shown mismatch")
        
        if isfield(PDS.data{iTrial}.pmBase, 'condsShown')
            numGratings = numel(PDS.data{iTrial}.pmBase.condsShown); % for some reason the number
        else
            continue
        end
        
        ft = PDS.data{iTrial}.timing.flipTimes(1,:);
        trialStart = ft(1);
        ft = ft - trialStart;
        
        for iGrating = 1:numGratings
            gratParams = mergeStruct(PDS.baseParams.(gfields{iGrating}), PDS.data{iTrial}.(gfields{iGrating}));
            
            assert(PDS.data{iTrial}.pmBase.condsShown(iGrating) == gratParams.condIndex, "condition mismatch")
            gratingOnset = ft(find(ft > gratParams.modOnDur(1), 1)) + trialStart;
            gratingOffset = ft(find(ft > gratParams.modOnDur(2), 1)) + trialStart;
            if isempty(gratingOffset)
                gratingOffset = ft(end) + trialStart;
            end
            
            D.GratingContrast = [D.GratingContrast; gratParams.contrast];
            D.GratingOnsets = [D.GratingOnsets; gratingOnset];
            D.GratingOffsets = [D.GratingOffsets; gratingOffset];
            D.GratingDirections = [D.GratingDirections; gratParams.dirs];
            D.GratingFrequency = [D.GratingFrequency; gratParams.grateSf];
            D.GratingSpeeds = [D.GratingSpeeds; gratParams.grateTf/gratParams.grateSf];
        end
        % treadmill time is the frame times
        D.treadTime = [D.treadTime; ft(:) + trialStart];
        % smooth with gaussian, differentiate, divide by timestep to get speed
        tpos = PDS.data{iTrial}.locationSpace(2,:);
        tpos = tpos - tpos(1);
        dxdt = diff(imgaussfilt(tpos, ip.Results.wheelsmoothing)) ./ diff(ft);
        % smooth again because of dropped frames
        spd = [0; dxdt(:)];
        
        D.treadSpeed = [D.treadSpeed; spd];
        D.treadPos = [D.treadPos; tpos(:)];
        
        assert(numel(D.treadTime)==numel(D.treadSpeed))
        assert(numel(D.treadTime)==numel(D.treadPos))
        
        pos = squeeze(PDS.data{1}.eyelink.posRawFrames)';
        nt = min(numel(ft),size(pos,1));
        D.eyeTime = [D.eyeTime; ft(1:nt)' + trialStart];
        D.eyePos = [D.eyePos; pos(1:nt,:)/PDS.baseParams.display.ppd];
    end
    
    % fake marmoV5 struct to find saccades
    Exp = struct();
    Exp.vpx.smo = [D.eyeTime, D.eyePos D.eyePos];
    
    % Saccade processing:
    % Perform basic processing of eye movements and saccades
    Exp = saccadeflag.run_saccade_detection_cloherty(Exp, ...
        'ShowTrials', false,...
        'accthresh', 2e4,...
        'velthresh', 10,...
        'velpeak', 10,...
        'isi', 0.02);
    
    D.eyeLabels = Exp.vpx.Labels;
    
    if any(isinf(D.GratingOnsets))
    keyboard
    end

    % convert times to ephys
    timingFields = {'treadTime', 'GratingOnsets', 'GratingOffsets', 'eyeTime'};
    for iTimingField = 1:numel(timingFields)
        D.(timingFields{iTimingField}) = PTB2EP(D.(timingFields{iTimingField}))/double(sample_rate);
    end
    
    if any(isinf(D.GratingOnsets))
    keyboard
    end
    
    % --- exclude negative times
    gratFields = {'GratingDirections', 'GratingFrequency', 'GratingOffsets', 'GratingOnsets', 'GratingSpeeds', 'GratingContrast'};
    iix = D.GratingOnsets < 0;
    for iField = 1:numel(gratFields)
       D.(gratFields{iField})(iix) = [];
    end
    
    

    
    eyeFields = {'eyeLabels', 'eyePos', 'eyeTime'};
    iix = D.eyeTime < 0;
    for iField = 1:numel(eyeFields)
       D.(eyeFields{iField})(iix,:) = [];
    end
    
    treadFields = {'treadTime', 'treadSpeed'};
    iix = D.treadTime < 0;
    for iField = 1:numel(treadFields)
       D.(treadFields{iField})(iix) = [];
    end
    
    
end
disp('Done importing session');


