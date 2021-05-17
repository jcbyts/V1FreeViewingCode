
%% read dataset.csv
meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');
data = readtable(meta_file);

%% loop over imported sessions and export the grating subspace info

imported = find(~isnan(data.numChannels));
for sess = imported(:)'
    
    try
        io.gratingSubspaceExport(sess);
    catch
        fprintf('Error session %d\n', sess)
    end 
    
end

%% subset of sessions
for sess=1:25 % Ellie sessions
    io.gratingSubspaceExport(sess, 'spike_sorting', 'kilowf')
end

%%
for sess = [32:33 45 56:57]
%     try
    io.gratingSubspaceExport(sess, 'spike_sorting', 'jrclustwf');
%     end
end

%% copy to server
% server_string = 'jcbyts@sigurros';
% output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/grating_subspace';

server_string = 'jake@bancanus';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/grating_subspace';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
data_dir = fullfile(data_dir, 'grating_subspace');
flist = dir(fullfile(data_dir, '*gratingsubspace.mat'));

%%
fname = arrayfun(@(x) x.name, flist, 'uni', 0);

command = 'scp ';
for iFile = 1:numel(fname)
    command = [command fullfile(data_dir, fname{iFile}) ' '];
end

command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname{:})

%%

for i = 1:numel(fname)
    tmp = load(fullfile(data_dir, fname{i}));
    tmp.spikes.csdReversal
%     assert(numel(unique(tmp.eyepos(:,1)))==size(tmp.eyepos,1))
%     sum(diff(tmp.grating.frameTime)<=0)
end