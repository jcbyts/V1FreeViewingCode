function gratingSubspaceExport(sessionId)
% Export session info for grating subspace analysis
% gratingSubspaceExport(sessionId)
% sessionId can be a number or a session tag (e.g., "logan_20200303")

% handle data csv
meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');
data = readtable(meta_file);

% this function will save a new column that says whether the grating
% subspace data has been exported
if ~ismember('GratingSubspace', data.Properties.VariableNames)
    data.GratingSubspace = false(size(data,1),1);
end

% load session
Exp = io.dataFactoryGratingSubspace(sessionId);

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
fname = fullfile(dataDir, strrep(Exp.FileTag, '.mat', '_gratingsubspace.mat'));
save(fname, '-v7', 'grating', 'dots', 'slist', 'spikes', 'eyepos')

% update meta table
Tag = strrep(Exp.FileTag, '.mat', '');
sessix = strcmp(data.Tag, Tag);
data.GratingSubspace(sessix) = true;
writetable(data, meta_file);


