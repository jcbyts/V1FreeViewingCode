function varargout = import_spike_sorting(DataFolder)
% Import spike sorting results
% sp = import_spike_sorting(DataFolder)
%
% Input:
%   DataFolder: full path to data
%
% Output:
%  sp [cell] Jude's spike sorting cell array
%
% If a sp-kilo.mat file exists, import_spike_sorting will return that file
% over all others. If it does not, then it wil run threshold spike sorting

%
% Check for Kilosort First
processedDir = dir(fullfile(DataFolder, '*processed*'));
shanksDir = dir(fullfile(DataFolder, '_shank*'));
processedDir = [processedDir; shanksDir];
if ~isempty(processedDir)
    assert(numel(processedDir)==1, 'import_spike_sorting: I don''t know how to handle multiple processed directories yet. You have to implement that')
    
    % find kilosort file in the processed directory
    kiloFile = dir(fullfile(DataFolder, processedDir.name, '*kilo*.mat'));
    if ~isempty(kiloFile)
        fprintf('Found kilosort file\n')
        sp = load(fullfile(DataFolder, processedDir.name, kiloFile(1).name));
        
        %**********************
        % convert kilosort spike struct to Jude's cell array
        osp = sp;  % save old
        cids = unique(sp.clu);
        N = numel(cids);
        sp = cell(1, N);
        for k = 1:N
            sp{k}.st = osp.st(osp.clu==cids(k));
        end
        
        if nargout > 0
            varargout{1} = sp;
        end
        
        if nargout > 1
            varargout{2} = osp;
        end
        return
    end
else 
    if input('No spikes found. Do you want to do threshold spike sorting? (1 or 0)')
        
    else
        error('import_spike_sorting: no spikes')
    end
end

% If no Kilosort, proceed with old code
% **** based on what you set, it runs spike sorting (eventually KiloSort)
% search data directory for sorted spike files, and use those if available
spfiles = dir(fullfile(DataFolder, '*.*_spk.mat'));

if (~isempty(spfiles))
    fprintf('Using .spk files present in data directory for spike times\n');
    disp('Counting flagged units ....');
    CN = 0;
    holdsp = cell(1,1);
    for zk = 1:size(spfiles,1)
        fname = fullfile(DataFolder, spfiles(zk).name);
        load(fname);
        if ~isempty(sp)
            for k = 1:size(sp,2)
                CN = CN + 1;
                holdsp{1,CN} = sp{1,k};
                holdsp{1,CN}.name = [fname,'_',sprintf('U%d',k)];
            end
        end
    end
    clear sp;
    sp = holdsp;
    fprintf('Total of %d units identified in sorted files\n',CN);
else
    %% ******** Call the Spike Sorting Script with parameters set *******
    SingleChannel = false;  % if false the specify shank layout
    ChNumber = NaN;   % if single channel, set here the number
    FullSpread = 0;   % analyze user selected channels
    if (0)
        ShankLayout = 'ShankA_32map.txt';  % shank layout name
        ChanNums = 32;  % if a shank recording, how many channels, 1 to N
    else
        if (0)
            SingleChannel = true;  % if false the specify shank layout
            ChNumber = 25;   % if single channel, set here the number
            FullSpread = 0;   % analyze user selected channels
        else
            ShankLayout = [];
            ChanNums = 64;
            FullSpread = 10;  % hash all channels and give composite map
        end
    end
    
    %% *******S
    Spike_Sorting_Script;   % otherwise use a simple thresholding scrip
end

if nargout > 0
    varargout{1} = sp;
end

if nargout > 1
    if exist('osp', 'var')
        varargout{2} = osp;
    else
        varargout{2} = [];
    end
end
end