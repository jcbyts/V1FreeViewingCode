function copy_spikes_from_server(sessionId, spikesTag)
% copy spikes file from the server
% Input:
%   sessionid 
%   spikesTag (default: kilo)

if nargin < 2 || isempty(spikesTag)
    spikesTag = 'kilo';
end

serverDir = getpref('FREEVIEWING', 'SERVER_DATA_DIR');
dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');

fvpath = fileparts(which('addFreeViewingPaths'));
data = readtable(fullfile(fvpath, 'Data', 'datasets.xls'));

if ischar(sessionId)
    sessionId = find(strcmp(data.Tag, sessionId));
end

if isnumeric(sessionId)
    thisSession = data(sessionId,:);
end

rootDir = strrep(thisSession.RootDirectory{1}, '/', filesep);
rawDir = strrep(thisSession.Directory{1}, '/', filesep);

fpath = fullfile(serverDir, rootDir, rawDir);
[~, sp] = io.import_spike_sorting(fpath, spikesTag);

processedFileName = sprintf('%s_%s.mat', thisSession.Tag{1}, spikesTag);

if ~exist(fullfile(dataPath, 'spikes'), 'dir')
    mkdir(fullfile(dataPath, 'spikes'));
end

fprintf('saving...')
save(fullfile(dataPath, 'spikes', processedFileName), '-v7.3', '-struct', 'sp')
fprintf(' Done\n')


