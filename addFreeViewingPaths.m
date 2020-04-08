function varargout = addFreeViewingPaths(user)
% set paths for FREEVIEWING projects
% this assumes you are running from the FREEVIEWING folder
% test git
if nargin < 1
    error('addFreeViewingPaths: requires a user argument')
end

switch user

    case 'jakesigur'
        % we need marmoview in the path for the stimulus objects to
        % regenerate properly
        marmoViewPath = '~/Repos/MarmoV5/';
        % we only need marmopipe to import raw data
        marmoPipePath = [];
        % where the data live
        dataPath = '~/Data/MitchellV1FreeViewing/';
        
        % processed data:
        setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', dataPath)

    case 'jakelaptop'
        % we need marmoview in the path for the stimulus objects to
        % regenerate properly
        marmoViewPath = '~/Documents/MATLAB/MarmoV5/';
        % we only need marmopipe to import raw data
        marmoPipePath = '~/Dropbox/Marmo Lab Website/PSA/Code/';
        % where the data live
        dataPath = '~/Dropbox/Projects/FreeViewing/Data';
        
        % processed data:
        setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', dataPath)
        setpref('FREEVIEWING', 'SERVER_DATA_DIR', '/Volumes/mitchelllab/Data/')
        
    case 'jakework'
        % we need marmoview in the path for the stimulus objects to
        % regenerate properly
        marmoViewPath = 'C:\Users\Jake\Documents\MATLAB\MarmoV5';
        % we need the full marmopipe / import paths
        marmoPipePath = 'C:\Users\Jake\Dropbox\Marmo Lab Website\PSA\Code';
        % where the data live
        dataPath = 'C:\Users\Jake\Dropbox\Projects\FreeViewing\Data';
        
        % set preferences for where the data live. Storing these paths as
        % preferences means we can recover them from any other function.
        % It's super useful.
        
        % Raw unprocessed data:
        setpref('FREEVIEWING', 'SERVER_DATA_DIR', 'Z:\Data\ForageMapping\')
        % processed data:
        setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', dataPath)
    
    case 'gravedigger'
        
        marmoViewPath = 'C:\Users\Jake\Documents\MarmoV5';
        % we only need marmopipe to import raw data
        marmoPipePath = 'C:\Users\Jake\Dropbox\Marmo Lab Website\PSA\Code';
        % where the data live
        dataPath = 'C:\Users\Jake\Documents\FreeViewingData';
        
        % processed data:
        setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', dataPath)
        setpref('FREEVIEWING', 'SERVER_DATA_DIR', 'Z:\Data')
        
        
    case 'dansigur'
        marmoViewPath = '~/Marmo/MarmoV5/';
        marmoPipePath = [];
        dataPath = '~/Marmo/Data';
        
    otherwise
        error('addFreeViewingPaths: I don''t know this user')
end

projectPath = fileparts(mfilename('fullpath'));
addpath(fullfile(projectPath, 'MatlabCode'))

if ~isempty(marmoPipePath)
    addpath(marmoPipePath)
    addMarmoPipe
end

addpath(marmoViewPath)
addpath(fullfile(marmoViewPath, 'SupportFunctions'))
addpath(fullfile(marmoViewPath, 'SupportData'))
addpath(fullfile(marmoViewPath, 'Settings'))

if nargout == 1
    varargout{1} = dataPath;
end


    