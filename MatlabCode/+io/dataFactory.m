function varargout = dataFactory(sessionId, dataPath)
% DATAFACTORY is a big switch statement. You pass in the ID of the sesion
% you want and it returns the processed data.
%
% Input:
%   Session [Number or ID string]
% Output: 
%   Exp [struct]
%
% Example:
%   Exp = io.dataFactory(5); % load the 5th session

sessionList = {'L2019119', ...
    'L20191120', ...
    'L20191121', ...
    'L20191122', ...
    'L20191202', ...
    'L20191203', ...
    'L20191204', ...
    'L20191205', ...
    'L20191206', ...
    'L20191209', ...
    'L20191226', ...
    'L20191231'};

if nargin < 2
    
    projectPrefs = getpref('FREEVIEWING');
    if isempty(projectPrefs)
        error('dataFactory: You need to run setpaths for you user first')
    end
    
    dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
end
    
if nargin < 1
    n = numel(sessionList);
    fprintf('No session id passed in. You can call dataFactory with an id number or string:\n')
    for i = 1:n
        fprintf('%d) %s\n', i, sessionList{i});
    end
    
    if nargout == 1
        varargout{1} = sessionList;
    end
    
    return
end

S.Latency = 8.3e-3; % delay from PTB time to actual monitor refresh
S.rect = [-10 -10 100 100]; % default gaze-centered ROI (pixels)

switch sessionId       
    case {-1, 'L181119'}
        S.processedFileName = 'L181119.mat';
        S.rawFilePath = 'Logan_2019-11-18_14-17-04_Neuronexus_D1';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;

    case {1, 'L20191119'}
        S.processedFileName = 'L20191120.mat';
        S.rawFilePath = 'Logan_2019-11-19_10-52-51_Neuronexus_D2';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
    
    case {2, 'L20191120'}
        S.processedFileName = 'L20191120.mat';
        S.rawFilePath = 'Logan_2019-11-20_10-54-11_NeuronexusD3';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
    
    case {3, 'L20191121'}
        S.processedFileName = 'L20191121.mat';
        S.rawFilePath = 'Logan_2019-11-21_10-31-34_NeuronexusD4';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
        
     case {4, 'L20191122'}
        S.processedFileName = 'L20191122.mat';
        S.rawFilePath = 'Logan_2019-11-22_11-30-43_NeuronexusD5';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
	
	case {5, 'L20191202'}
        S.processedFileName = 'L20191202.mat';
        S.rawFilePath = 'Logan_2019-12-02_10-29-07_NeuronexusD6';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
    
    case {6, 'L20191203'}
        S.processedFileName = 'L20191203.mat';
        S.rawFilePath = 'Logan_2019-12-03_10-35-59_NeuronexusD7';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
        
    case {7, 'L20191204'}
        S.processedFileName = 'L20191204.mat';
        S.rawFilePath = 'Logan_2019-12-04_10-32-29_NeuronexusD8';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
        
    case {8, 'L20191205'}
        S.processedFileName = 'L20191205.mat';
        S.rawFilePath = 'Logan_2019-12-05_10-35-13_NeuronexusD9';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
        S.cids = [1:9, 12:15, 17:20, 21:22, 25:37, 39:41];
        S.rect = [-10 0 30 40];
    case {9, 'L20191206'}
        S.processedFileName = 'L20191206.mat';
        S.rawFilePath = 'Logan_2019-12-06_11-41-05_NeuronexusD10';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
        
    case {10, 'L20191209'}
        S.processedFileName = 'L20191209.mat';
        S.rawFilePath = 'Logan_2019-12-09_10-38-50_NeuronexusD11';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 1;
        
    case {11, 'L20191226'}
        S.processedFileName = 'L20191226.mat';
        S.rawFilePath = 'Logan_2019-12-02_10-29-07_NeuronexusD16';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 2;
        
    case {12, 'L20191231'}
        S.processedFileName = 'L20191231.mat';
        S.rawFilePath = 'Logan_2019-12-02_10-29-07_NeuronexusD19';
        S.importFun = @io.importFreeViewing;
        S.probeNum = 3;
        S.rect = [-20 0 20 40];

    otherwise
        error('dataFactory: unrecognized session')
end

% try loading the file
fname = fullfile(dataPath, S.processedFileName);
if exist(fname, 'file')
    fprintf('Loading [%s]\n', S.processedFileName)
    Exp = load(fname);
    fprintf('Done\n')
else % try importing the file
    fprintf('Could not find [%s]\n', fname)
    fprintf('Trying to import the data from [%s]\n', S.rawFilePath)
    
    % incase the import function changes to match changes in the
    % dataformat, this is a function handle that can be specific to certain
    % functions
    Exp = S.importFun(S); 
    
    
    % --- Save Exp Struct to data folder for future use (PROC folder)
    % should consider later how we want to store things (per protocol,
    % but off hand it seems feasible given the generic structure of
    % of the D struct maybe we could concatenate all sessions in one
    Exp.ProcDataFolder = dataPath;
    Exp.DataFolder = getpref('FREEVIEWING', 'SERVER_DATA_DIR');
    Exp.FileTag = S.processedFileName;
    save(fname,'-v7.3', '-struct', 'Exp');
    fprintf('Exp struct saved to %s\n',fname);
    fprintf('Mat File %s\n',S.processedFileName);
end

varargout{1} = Exp;

if nargout > 1
    if ~isfield(S, 'cids')
        S.cids = Exp.osp.cids;
    end

    varargout{2} = S;
end
    