

%% get JRCLUST spikes
for iEx = 5

[~, S] = io.dataFactoryGratingSubspace(iEx);

fpath = S.rawFilePath;
ops = io.loadOps(fpath);
ops = io.convertOpsToNewDirectory(ops, fpath);

sp = io.getSpikesFromJRCLUST(ops);

io.copy_spikes_from_server(iEx, 'JRC');

%% reload Exp and clip waveforms from hp-filtered traces
[Exp, S] = io.dataFactoryGratingSubspace(iEx, 'spike_sorting', 'JRC', 'cleanup_spikes', 1);

io.clip_waveforms(Exp, S, 'Fs', 30e3);

end

%% Gather all folders that have prm files
dataPath = 'Z:\Data\ForageMapping';

flist = dir(dataPath);
flist(1:2) = [];
flist(~arrayfun(@(x) x.isdir, flist)) = [];

nfolders = numel(flist);
prmFiles = {};
fprintf('Search through %d folders in [%s]\n', nfolders, dataPath)
for f = 1:nfolders
    fprintf('%d) %s\n', f, flist(f).name)
    
    if ~flist(f).isdir
        warning('not a directory. skipping')
        continue
    end
    
    fpath = dir(fullfile(dataPath, flist(f).name, '*_processed*'));
    
    if isempty(fpath)
        warning('no processed folder. skipping')
        continue
    end
    
    prmfile = fullfile(dataPath, flist(f).name, fpath(1).name, 'ephys-jr.prm');
    if exist(prmfile, 'file') == 0
        warning('no JRCLUST params file. skipping')
        continue
    end
    
    prmFiles{f} = prmfile;
        
end

%%

prmFiles(cellfun(@isempty, prmFiles)) = [];

for i = 1:numel(prmFiles)
    fprintf('%d) %s\n', i, prmFiles{i});
end

%%

iSess = 12;
prmf = prmFiles{iSess};

%%
prmf = 'Z:\Data\PLDAPS\Ellie\Ellie_2017-07-31_15-21-11_Shnkd8\_shank01_V1\ephys-jr.prm';
jrc('preview', prmf)

%%
jrc('detect-sort', prmf)

%%
jrc('manual', prmf)


