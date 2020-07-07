function stats = dataFactoryCSD(sessionId, varargin)
% DATAFACTORY is a big switch statement. You pass in the ID of the sesion
% you want and it returns the processed data.
%
% Input:
%   Session [Number or ID string]
%   type: either 'csd' or 'gamma'
% Output: 
%   stats [struct]
%
% Example:
%   stats = io.dataFactoryCSD(5, 'type', 'csd'); % load the 5th session CSD



ip = inputParser();
ip.addParameter('type', 'csd')
ip.parse(varargin{:});

type = ip.Results.type;

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

S.processedFileName = [thisSession.Tag{1} '.mat'];

switch type
    case 'csd'
        fname = fullfile(dataPath, 'csd', S.processedFileName);
        % try loading the file
        if exist(fname, 'file')
            fprintf('Loading [%s]\n', S.processedFileName)
            stats = load(fname);
        else
            fprintf('Could not find CSD for [%s]\n', fname)
            fprintf('Generating CSD to save in [%s]\n', fname)
            [Exp, ~, lfp] = io.dataFactoryGratingSubspace(sessionId);
            stats = csd.getCSD(lfp, Exp, 'plot', false);
            save(fname,'-v7.3', '-struct', 'stats')
            disp('Done')
        end
    case 'gamma'
        fname = fullfile(dataPath, 'gamma', S.processedFileName);
        % try loading the file
        if exist(fname, 'file')
            fprintf('Loading [%s]\n', S.processedFileName)
            stats = load(fname);
        else
            fprintf('Could not find gamma for [%s]\n', fname)
            fprintf('Generating gamma to save in [%s]\n', fname)
            [~, ~, lfp] = io.dataFactoryGratingSubspace(sessionId);
            stats = csd.getGamma(lfp, 'plotIt', false);
            save(fname,'-v7.3', '-struct', 'stats')
            disp('Done')
        end
        
        
end



end

