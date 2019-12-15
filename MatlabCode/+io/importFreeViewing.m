function Exp = importFreeViewing(S)
% this function imports the data required for the FREEVIEWING project
% It shares the same fundamentals as 
disp('FREE VIEWING IMPORT')
disp('THIS MUST BE RUN ON A COMPUTER CONNECTED TO THE MITCHELLLAB SERVER')
disp('REQUIRES MARMOPIPE CODE IN THE PATH')

% get paths
SERVER_DATA_DIR = getpref('FREEVIEWING', 'SERVER_DATA_DIR');

DataFolder = fullfile(SERVER_DATA_DIR, S.rawFilePath);

assert(exist(DataFolder, 'dir')==7, 'importFreeViewing: raw data path does not exist')

% Load spikes data
[sp,osp] = io.import_spike_sorting(DataFolder);

% Baic marmoView import. Synchronize with Ephys if it exists
Exp = io.basic_marmoview_import(DataFolder);

Exp.sp = sp;  % Jude's cell array of spike times
Exp.osp = osp; % keep the original Kilosort spike format

% Import eye position signals
Exp = io.import_eye_position(Exp, DataFolder);

% Saccade processing:
% Perform basic processing of eye movements and saccades
Exp = saccadeflag.run_saccade_detection(Exp, 'ShowTrials', false);

disp('Done importing session');

