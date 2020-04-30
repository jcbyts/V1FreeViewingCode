
%% read dataset.csv
meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');
data = readtable(meta_file);

%% loop over imported sessions and export the grating subspace info

imported = find(~isnan(data.numChannels));
for sess = imported(:)'
    try
        io.gratingSubspaceExport(sess);
    end
end

%