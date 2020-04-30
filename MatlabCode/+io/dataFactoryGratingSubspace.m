function varargout = dataFactoryGratingSubspace(sessionId)
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

dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');

meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');
% meta_file = fullfile(dataPath, 'datasets.xls');

data = readtable(meta_file);
nSessions = size(data,1);

if nargin < 1
    sessionId = [];
end

if ischar(sessionId)
    sessionId = find(strcmp(data.Tag, sessionId));
end

if isempty(sessionId)
    fprintf('No valid session id passed in. You can call dataFactory with an id number or string:\n')
    for i = 1:nSessions
        fprintf('%d) %s\n', i, data.Tag{i})
    end
    return
end
    
if isnumeric(sessionId)
    thisSession = data(sessionId,:);
end
    

S.Latency = 8.3e-3; % delay from PTB time to actual monitor refresh
S.rect = [-10 -10 100 100]; % default gaze-centered ROI (pixels)

S.processedFileName = [thisSession.Tag{1} '.mat'];

rootDir = strrep(thisSession.RootDirectory{1}, '/', filesep);
rawDir = strrep(thisSession.Directory{1}, '/', filesep);

% try loading the file
fname = fullfile(dataPath, S.processedFileName);
if exist(fname, 'file')
    fprintf('Loading [%s]\n', S.processedFileName)
    Exp = load(fname);
    fprintf('Done\n')
else % try importing the file
    serverDir = getpref('FREEVIEWING', 'SERVER_DATA_DIR');

    S.rawFilePath = fullfile(serverDir, rootDir, rawDir);
    S.importFun = str2func(['io.' thisSession.ImportFun{1}]);
    fprintf('Could not find [%s]\n', fname)
    fprintf('Trying to import the data from [%s]\n', S.rawFilePath)
    
    % incase the import function changes to match changes in the
    % dataformat, this is a function handle that can be specific to certain
    % functions
    Exp = S.importFun(S); 
    
    % some more meta data
    ops = io.loadOps(S.rawFilePath);
    S.numChan = sum([ops.Nchan]);
    S.numSU = sum(Exp.osp.isiV<.2);
    S.numU = numel(Exp.osp.isiV);
    
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
    
    % --- save meta data to dataset table
    fprintf('Updating Dataset Table\n')
    thisSession.numChannels = S.numChan;
    thisSession.numUnits = S.numU;
    thisSession.numSingleUnit = S.numSU;
    
    data(sessionId,:) = thisSession;
    writetable(data, meta_file);
    fprintf('Done\n')
end

varargout{1} = Exp;

if nargout > 1
    if ~isfield(S, 'cids')
        S.cids = Exp.osp.cids;
    end

    varargout{2} = S;
    
    if nargout > 2 % get LFP
        disp('getting Local Field Potentials')
        fname = fullfile(dataPath, 'lfp', S.processedFileName);
        
        fprintf('Loading [%s]\n', S.processedFileName)
        if exist(fname, 'file')
            lfp = load(fname);
            fprintf('Done\n')
        else % try importing the file
            serverDir = getpref('FREEVIEWING', 'SERVER_DATA_DIR');

            S.rawFilePath = fullfile(serverDir, rootDir, rawDir);            
            fprintf('Could not find LFP file for [%s]\n', fname)
            fprintf('Trying to import the data from [%s]\n', S.rawFilePath)
    
            % some more meta data
            ops = io.loadOps(S.rawFilePath);
            load(ops.chanMap);
            [data, timestamps] = io.getLFP(ops);
            if ~isfolder(fullfile(dataPath, 'lfp'))
                mkdir(fullfile(dataPath, 'lfp'))
            end
            save(fname, '-v7', 'timestamps', 'data', 'xcoords', 'ycoords', 'zcoords') % v7 flag can be read easily by scipy
            lfp = struct('timestamps', timestamps, 'data', data, 'xcoords', xcoords, 'ycoords', ycoords, 'zcoords', zcoords);
        end
        
        varargout{3} = lfp;
        
    end
end
    