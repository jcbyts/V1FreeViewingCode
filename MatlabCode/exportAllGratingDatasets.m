
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
% for sess = 1:57
%     try
%     io.gratingSubspaceExport(sess);
%     end
% end

%% copy to server
server_string = 'jcbyts@sigurros';
output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
data_dir = fullfile(data_dir, 'grating_subspace');
flist = dir(fullfile(data_dir, '*gratingsubspace.mat'));
fname = arrayfun(@(x) x.name, flist, 'uni', 0);

command = 'scp ';
for iFile = 1:numel(fname)
    command = [command fullfile(data_dir, fname{iFile}) ' '];
end

command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname{:})