function gratingSubspaceExport(sessionId)
% Export session info for grating subspace analysis
% gratingSubspaceExport(sessionId)
% sessionId can be a number or a session tag (e.g., "logan_20200303")

[Exp, S] = io.dataFactoryGratingSubspace(sessionId);

%% get relevant 
[~, ~, grating] = io.preprocess_grating_subspace_data(Exp);
[~, ~, dots] = io.preprocess_spatialmapping_data(Exp);

spikes = Exp.osp;

slist = Exp.slist;
for i = 1:3
    slist(:,i) = Exp.vpx2ephys(slist(:,i));
end

eyepos = Exp.vpx.smo(:,1:3);
eyepos(:,1) = Exp.vpx2ephys(eyepos(:,1));

dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
dataDir = fullfile(dataDir, 'grating_subspace');

if ~exist(dataDir, 'dir')
    mkdir(dataDir)
end
fname = fullfile(dataDir, strrep(Exp.FileTag, '.mat', '_gratingsubspace.mat'));
save(fname, '-v7', 'grating', 'dots', 'slist', 'spikes', 'eyepos')