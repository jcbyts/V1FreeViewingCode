clear all
%%
addpath ~/'Google Drive'/HuklabTreadmill/PLDAPStools/

% --- get path
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
subj = 'gru';

% --- get concatenated spikes file
spikesfname = fullfile(fpath, subj, [subj '_All_cat.mat']);
fprintf('Loading spikes from [%s]\n', spikesfname)
EphysData = load(spikesfname);

% --- find ephys and behavioral date tags
ephysSessions = EphysData.z.RecId';
behaveSessions = io.dataFactoryTreadmillSS';

% --- convert dates to number and matsch sessions
behaveDates = cellfun(@(x) datenum(x(numel(subj)+2:end), 'yyyymmdd'), behaveSessions);
ephysDates = cellfun(@(x) datenum(x(numel(subj)+2:numel(subj)+11), 'yyyy-mm-dd'), ephysSessions);

[sessionMatch, behave2ephysNum] = ismember(ephysDates, behaveDates);

goodEphysSessions = find(sessionMatch);

%% --- loop over sessions, build big spike struct

Dbig = struct('GratingDirections', [], 'GratingFrequency', [], ...
    'GratingOffsets', [], 'GratingOnsets', [], 'GratingSpeeds', [], ...
    'eyeLabels', [], 'eyePos', [], 'eyeTime', [], ...
    'treadSpeed', [], 'treadTime', [], ...
    'sessNumSpikes', [], 'sessNumGratings', [], 'spikeTimes', [], 'spikeIds', []);

fprintf('Loading and aligning spikes with behavior\n')

startTime = 0; % all experiments start at zero. this number will increment as we concatenate sessions
newOffset = 0;
timingFields = {'GratingOnsets', 'GratingOffsets', 'spikeTimes', 'treadTime', 'eyeTime'};
nonTimingFields = {'GratingDirections', 'GratingFrequency', 'GratingSpeeds', 'eyeLabels', 'eyePos', 'treadSpeed', 'spikeIds', 'sessNumSpikes', 'sessNumGratings'};


for iSess = 1:numel(goodEphysSessions)
    
    rId = goodEphysSessions(iSess);
    bSess = behave2ephysNum(goodEphysSessions(iSess));
    
    fprintf('Session %d/%d [ephys:%d , behav:%d]\n', iSess, numel(goodEphysSessions), rId, bSess)
    
    unitlist = find(cellfun(@(x) ~isempty(x), EphysData.z.Times{rId}));

    st = [];
    clu = [];

    for iunit = 1:numel(unitlist)
        kunit = unitlist(iunit);
    
        stmp = double(EphysData.z.Times{rId}{kunit}) / EphysData.z.Sampling;
        st = [st; stmp];
        clu = [clu; kunit*ones(numel(stmp),1)];
    end
    
    D_ = io.dataFactoryTreadmillSS(bSess);
    D_.spikeTimes = st;
    D_.spikeIds = clu;
    
    % convert to D struct format
    D = io.get_drifting_grating_output(D_);
    D.sessNumSpikes = iSess*ones(size(D.spikeTimes));
    D.sessNumGratings = iSess*ones(size(D.GratingOnsets));
    
    % loop over timing fields and offset time
    for iField = 1:numel(timingFields)
        Dbig.(timingFields{iField}) = [Dbig.(timingFields{iField}); D.(timingFields{iField}) + startTime];
        newOffset = max(newOffset, max(Dbig.(timingFields{iField}))); % track the end of this session
    end
    
    for iField = 1:numel(nonTimingFields)
        Dbig.(nonTimingFields{iField}) = [Dbig.(nonTimingFields{iField}); D.(nonTimingFields{iField})];
    end
    
    startTime = newOffset + 2; % 2 seconds between sessions
    if isinf(startTime)
        keyboard
    end
        
    fprintf('StartTime: %02.2f\n', startTime)
end


%% Save

figure(1); clf
iix = Dbig.treadSpeed > 200;
treadSpeed = Dbig.treadSpeed;
treadSpeed(iix) = nan;
iix = diff(treadSpeed).^2 > 50;
treadSpeed(iix) = nan;

plot(treadSpeed)

fprintf('Saving... ')
save(fullfile(fpath, 'gruD_all.mat'), '-v7.3', '-struct', 'Dbig')
disp('Done')



