

flist = dir('Data/*hdf5');

for f = 5%:numel(flist)
    finfo = h5info(fullfile(flist(f).folder, flist(f).name));
    fprintf('File [%s]:\n', flist(f).name)
    fprintf('\t%s\n', finfo.Groups.Name)
    fprintf('\n\n')
    
    for g1 = 1:numel(finfo.Groups)
        
        for g2 = 1:numel(finfo.Groups(g1).Groups)
            if any(strcmp({finfo.Groups(g1).Groups(g2).Datasets.Name}, 'blocks'))
                fprintf('Found blocks in [%s]\n', finfo.Groups(g1).Groups(g2).Name)
            end
        end
    end
end

%% copy to server

inds = 6;
for i = inds(:)'
    fname = fullfile(flist(i).folder, flist(i).name);
    server_string = 'jake@bancanus'; %'jcbyts@sigurros';
    output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';
    
    data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
    command = 'scp ';
    command = [command fname ' '];
    command = [command server_string ':' output_dir];
    
    system(command)
%     fprintf('%s\n', command)
    fprintf('%s\n', flist(i).name)
end