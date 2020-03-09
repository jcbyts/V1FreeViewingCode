function gratingSubspaceExport(sessionId)
% Export session info for grating subspace analysis
% gratingSubspaceExport(sessionId)
% sessionId can be a number or a session tag (e.g., "logan_20200303")

[Exp, S] = io.dataFactoryGratingSubspace(sessionId);

%% get relevant 
[stim, Robs, opts, params, oris, spatfreq] = io.preprocess_grating_subspace_data(Exp, 'stim_field', 'Grating');

slist = Exp.slist;
stim = stim{1};
frameTimes = opts.frameTime;

dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
dataDir = fullfile(dataDir, 'grating_subspace');
if ~exist(dataDir, 'dir')
    mkdir(dataDir)
end
fname = fullfile(dataDir, strrep(Exp.FileTag, '.mat', '_gratingsubspace.mat'));
save(fname, '-v7', 'stim', 'Robs', 'slist', 'frameTimes', 'oris', 'spatfreq', 'opts')