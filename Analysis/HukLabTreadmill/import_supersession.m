function import_supersession(subj, fpath)
% Create a Super Session file for a particular subject

% --- get path
if ~exist('fpath', 'var')
    fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
    fpath = fullfile(fpath, 'gratings');
end

% addpath(fullfile(fpath, 'PLDAPStools')) % add PLDAPStools

if nargin < 1
    subj = 'gru';
end

validSubjs = {'gru', 'brie'};
assert(ismember(subj,validSubjs), sprintf("import_supersession: subj name %s is not valid", subj))

%% --- loop over sessions, build big spike struct

Dbig = struct('GratingDirections', [], 'GratingFrequency', [], ...
    'GratingOffsets', [], 'GratingOnsets', [], 'GratingSpeeds', [], ...
    'GratingContrast', [], ...
    'eyeLabels', [], 'eyePos', [], 'eyeTime', [], ...
    'frameTimes', [], 'framePhase', [], 'frameContrast', [], ...
    'treadSpeed', [], 'treadTime', [], ...
    'sessNumSpikes', [], 'sessNumGratings', [], ...
    'sessNumTread', [], 'sessNumEye', [], 'spikeTimes', [], 'spikeIds', []);

Dbig.units = {};
%%

fprintf('Loading and aligning spikes with behavior\n')

unique_sessions = {'gru_20211217', 'allen'};

flist = dir(fullfile(fpath, [subj '*']));
startTime = 0; % all experiments start at zero. this number will increment as we concatenate sessions
newOffset = 0;
timingFields = {'GratingOnsets', 'GratingOffsets', 'spikeTimes', 'treadTime', 'eyeTime', 'frameTimes'};
nonTimingFields = {'GratingDirections', 'GratingFrequency', 'GratingSpeeds', 'eyeLabels', 'eyePos', 'treadSpeed', 'spikeIds', 'sessNumSpikes', 'sessNumGratings', 'sessNumTread', 'sessNumEye', 'framePhase', 'GratingContrast', 'frameContrast'};

fprintf('Looping over %d sessions\n', numel(flist))

for iSess = 1:numel(flist)
    D = load(fullfile(flist(iSess).folder, flist(iSess).name));


    % fix any wierdness from scipy
    fields = fieldnames(D);
    for f = 1:numel(fields)
        fprintf('[%s]\n', fields{f})
        if strcmp(fields{f}, 'unit_area')
            sz = size(D.unit_area);
            unit_area = cell(sz(1), 1);
            for i = 1:sz(1)
                unit_area{i} = strrep(D.unit_area(i,:), ' ', '');
            end
            D.unit_area = unit_area;
            continue
    
        end
    
        if iscell(D.(fields{f}))
            isnull = strcmp(D.(fields{f}), 'null');
            tmp = cellfun(@double, D.(fields{f}), 'uni', 0);
            tmp = cellfun(@(x) x(1), tmp);
            tmp(isnull) = nan;
            D.(fields{f}) = tmp;
        end
    end

    if isfield(D, 'unit_area')
        cids = unique(D.spikeIds);
        visp = strcmp(D.unit_area, 'VISp');
        
        iix = ismember(D.spikeIds, cids(visp));
        D.spikeTimes = D.spikeTimes(iix);
        D.spikeIds = D.spikeIds(iix);
    end

    if min(size(D.frameTimes)) > 1
        D.frameTimes = D.frameTimes(:);
        D.framePhase = D.framePhase(:);
        D.frameContrast = D.frameContrast(:);
        [D.frameTimes, ind] = sort(D.frameTimes);
        D.framePhase = D.framePhase(ind);
        D.frameContrast = D.frameContrast(ind);
    end
    
    if all(isnan(D.eyePos(:)))
        D.eyePos = nan(numel(D.frameTimes), 3);
        D.eyeTime = D.frameTimes;
        D.eyeLabels = ones(numel(D.frameTimes),1);
    end

    if any(cellfun(@(x) contains(flist(iSess).name, x), unique_sessions))
        
        unit_offset = max(unique(Dbig.spikeIds));
        if isempty(unit_offset)
            unit_offset = 0;
        end
        fprintf('offsetting spike ID by %d\n', unit_offset)
        D.spikeIds = D.spikeIds + unit_offset;
        if isfield(D, 'units')
            for ii = 1:numel(D.units)
                D.units(ii).id = D.units(ii).id + unit_offset;
            end
        end
    end
    
    if isempty(D.frameContrast)
        continue
    end
    
    D.sessNumSpikes = iSess*ones(size(D.spikeTimes));
    D.sessNumGratings = iSess*ones(size(D.GratingOnsets));
    D.sessNumTread = iSess*ones(size(D.treadTime));
    D.sessNumEye = iSess*ones(size(D.eyeTime));

    
    sessStart = 0;
    for iField = 1:numel(timingFields)
        tmp = min(sessStart, min(reshape(D.(timingFields{iField}), [], 1)));
        if isempty(tmp)
            continue
        end
    end
    
    % loop over timing fields and offset time
    for iField = 1:numel(timingFields)
        Dbig.(timingFields{iField}) = [Dbig.(timingFields{iField}); D.(timingFields{iField}) - sessStart + startTime];
        newOffset = max(newOffset, max(Dbig.(timingFields{iField}))); % track the end of this session
    end
    
    for iField = 1:numel(nonTimingFields)
        tmp = D.(nonTimingFields{iField});
        Dbig.(nonTimingFields{iField}) = [Dbig.(nonTimingFields{iField}); tmp];
    end
    
    startTime = newOffset + 2; % 2 seconds between sessions
    if isinf(startTime)
        keyboard
    end

    assert(numel(unique(D.spikeIds))==numel(D.units), 'import_supersession: mismatch in number of units on session')
    for cc = 1:numel(D.units)
        if isfield(D.units(cc), 'area') && ~isnan(D.units(cc).area)
            if numel(Dbig.units) < D.units(cc).id
                Dbig.units{D.units(cc).id} = {D.units(cc)};
            else
                if isempty(Dbig.units{D.units(cc).id})
                    Dbig.units{D.units(cc).id} = {D.units(cc)};
                else
                    Dbig.units{D.units(cc).id} = [Dbig.units{D.units(cc).id} {D.units(cc)}];
                end
            end
        end
    end

    fprintf('StartTime: %02.2f\n', startTime)
end


%% Save

iix = Dbig.treadSpeed > 200;
treadSpeed = Dbig.treadSpeed;
treadSpeed(iix) = nan;
iix = diff(treadSpeed).^2 > 50;
treadSpeed(iix) = nan;
% treadSpeed = repnan(treadSpeed, 'pchip'); % interpolate between artifacts
Dbig.treadSpeed = treadSpeed;


fprintf('Saving... ')
save(fullfile(fpath, [subj 'D_all.mat']), '-v7.3', '-struct', 'Dbig')
disp('Done')



