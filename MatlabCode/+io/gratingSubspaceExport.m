function gratingSubspaceExport(sessionId, varargin)
% Export session info for grating subspace analysis
% gratingSubspaceExport(sessionId)
% sessionId can be a number or a session tag (e.g., "logan_20200303")
% 'ellie_20181223'
% 'ellie_20181224'
csdExclusionList = {'ellie_20190107', ...
    'ellie_20190111', ...
    'logan_20191119', ...
    'logan_20191121', ...
    'logan_20191209', ...
    'logan_20200228', ...
    'logan_20200229', ...
    'logan_20200302', ...
    'milo_20190607', ...
    'milo_20190621'};

ip = inputParser();
ip.addParameter('spike_sorting', 'jrclustwf')
ip.addParameter('cleanup_spikes', 0)
ip.parse(varargin{:});
    
% handle data csv
meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');
data = readtable(meta_file);

% this function will save a new column that says whether the grating
% subspace data has been exported
if ~ismember('GratingSubspace', data.Properties.VariableNames)
    data.GratingSubspace = false(size(data,1),1);
end

% load session
[Exp,~,lfp] = io.dataFactoryGratingSubspace(sessionId, 'spike_sorting', ip.Results.spike_sorting, 'cleanup_spikes', ip.Results.cleanup_spikes);

if isempty(lfp.deadChan)
    v = var(lfp.data);

    thresh = median(v)*.5;
    deadChan = find(v < thresh);
    lfp.deadChan = deadChan;
end


if any(strcmp(csdExclusionList, Exp.FileTag(1:end-4)))
    csdReversal = nan;
else
    try
        et = csd.getCSDEventTimes(Exp);
        cstruct = csd.getCSD(lfp, et);
        csdReversal = nanmean(cellfun(@(x) x(1), cstruct.reversalPointDepth));
    catch
        csdReversal = nan;
    end
end


% get subspace data
[~, ~, grating] = io.preprocess_grating_subspace_data(Exp);
[~, ~, dots] = io.preprocess_spatialmapping_data(Exp);

% spikes
spikes = Exp.osp;

% saccades
slist = Exp.slist;
for i = 1:3
    slist(:,i) = Exp.vpx2ephys(slist(:,i));
end

% eye position
eyepos = Exp.vpx.smo(:,1:3);
eyepos(:,1) = Exp.vpx2ephys(eyepos(:,1));
eyepos = [eyepos Exp.vpx.Labels];

% output path
dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
dataDir = fullfile(dataDir, 'grating_subspace');

if ~exist(dataDir, 'dir')
    mkdir(dataDir)
end


% update meta table
Tag = strrep(Exp.FileTag, '.mat', '');
sessix = strcmp(data.Tag, Tag);

% get visual units stats
evalc('[visUnits,W] = io.get_visual_units(Exp, ''plotit'', false, ''visStimField'', ''Grating'');');

rf.isviz = arrayfun(@(x) double(x.Grating), visUnits);
rf.srf = reshape(cell2mat(arrayfun(@(x) x.srf(:)', visUnits, 'uni', 0))', [size(visUnits(1).srf) numel(visUnits)]);
rf.xax = visUnits(1).xax;
rf.yax = visUnits(1).yax;
rf.mu = [data.retx(sessix), data.rety(sessix)];
rf.cov = [data.retc1(sessix), data.retc2(sessix); data.retc2(sessix) data.retc4(sessix)];

if isfield(spikes, 'wfs')
    spikes = rmfield(spikes, 'wfs');
    spikes = rmfield(spikes, 'wftax');
end

spikes.peakMinusTrough = arrayfun(@(x) x.peaktime-x.troughtime, W);
spikes.isiRate = arrayfun(@(x) x.isiRate, W);
spikes.localityIdx = arrayfun(@(x) x.localityIdx, W);
spikes.csdReversal = csdReversal;
spikes.isi = cell2mat(arrayfun(@(x) x.isi, W, 'uni', 0));
spikes.isifit = cell2mat(arrayfun(@(x) x.isifit(:)', W, 'uni', 0));
spikes.isilags = cell2mat(arrayfun(@(x) x.lags, W, 'uni', 0));

fname = fullfile(dataDir, strrep(Exp.FileTag, '.mat', '_gratingsubspace.mat'));
save(fname, '-v7.3', 'grating', 'dots', 'slist', 'spikes', 'eyepos', 'rf')

data.GratingSubspace(sessix) = true;
writetable(data, meta_file);


