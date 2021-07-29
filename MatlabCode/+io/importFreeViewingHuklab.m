function Exp = importFreeViewingHuklab(S)
% this function imports the data required for the FREEVIEWING project
% It shares the same fundamentals as 
disp('FREE VIEWING IMPORT FOR HUKLAB DATASHARE')
disp('THIS MUST BE RUN ON A COMPUTER CONNECTED TO THE HUKLAB DATASHARE DRIVE FOLDER')
disp('REQUIRES MARMOPIPE CODE IN THE PATH')

% get paths
SERVER_DATA_DIR = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');

if contains(S.rawFilePath, SERVER_DATA_DIR)
    DataFolder = S.rawFilePath;
else
    DataFolder = fullfile(SERVER_DATA_DIR, S.rawFilePath);
end

assert(exist(DataFolder, 'dir')==7, 'importFreeViewing: raw data path does not exist')

% make folder for exporting import quality checks
datashare = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
exname = strrep(S.processedFileName, '.mat', '');

figDir = fullfile(datashare, 'imported_sessions_qa', exname);
if ~exist(figDir, 'dir')
    mkdir(figDir)
end

fpath = fullfile(figDir, 'import.txt');
fid = fopen(fpath, 'wb');

% %% Load spikes data
% [sp,osp] = io.import_spike_sorting(DataFolder, S.spikeSorting);
% io.copy_spikes_from_server(strrep(S.processedFileName, '.mat', ''), S.spikeSorting)

%% Baic marmoView import. Synchronize with Ephys if it exists
Exp = io.basic_marmoview_import(DataFolder, 'fid', fid);

% fix spatial frequency bug: marmoV5 data before 2021 February had a
% scaling bug where all spatial frequencies off by a factor of two
datebugfixed = datenum('20210201', 'yyyymmdd');
thisdate = datenum(regexp(S.processedFileName, '[0-9]*', 'match', 'once'), 'yyyymmdd');
if thisdate < datebugfixed
%     warning('importFreeViewing: fixing early MarmoV5 spatial frequency bug')
    fprintf(fid, 'importFreeViewing: fixing early MarmoV5 spatial frequency bug');
    Exp = io.correct_spatfreq_by_half(Exp);
end

% cleanup eye position files
edfFiles = dir(fullfile(DataFolder, '*edf.mat'));
for iFile = 1:numel(edfFiles)
    edfname = fullfile(edfFiles(iFile).folder, edfFiles(iFile).name);
    Dtmp = load(edfname);
    Dtmp.edf = Dtmp.e;
    save(edfname, '-v7.3', '-struct', 'Dtmp');
end

% Import eye position signals
[Exp, fig] = io.import_eye_position(Exp, DataFolder);
if ~isempty(fig)
    saveas(fig, fullfile(figDir, 'eye_time_sync.pdf'))
end

fig = io.checkCalibration(Exp);
saveas(fig, fullfile(figDir, 'eye_calibration_from_file.pdf'))

% redo calibration offline using FaceCal data
eyePos = io.getEyeCalibrationFromRaw(Exp);
Exp.vpx.smo(:,2:3) = eyePos;

fig = io.checkCalibration(Exp);
saveas(fig, fullfile(figDir, 'eye_calibration_redo_auto.pdf'))

% 
%%

% trial = 15;
% eyepos = Exp.D{trial}.eyeData(:,2:3);
% eyepos = Exp.vpx.raw(:,2:3);
% eyepos(:,2) = 1 - eyepos(:,2);
% 
% x = (eyepos(:,1)-Exp.D{trial}.c(1)) / (Exp.D{trial}.dx*Exp.S.pixPerDeg);
% y = (eyepos(:,2)-Exp.D{trial}.c(2)) / (Exp.D{trial}.dy*Exp.S.pixPerDeg);
% 
% figure(1); clf
% plot(x, y)


%%
% 


% Saccade processing:
% Perform basic processing of eye movements and saccades
Exp = saccadeflag.run_saccade_detection_cloherty(Exp, ...
    'ShowTrials', false,...
    'accthresh', 2e4,...
    'velthresh', 10,...
    'velpeak', 10,...
    'isi', 0.02);

% track invalid sampls
Exp.vpx.Labels(isnan(Exp.vpx.raw(:,2))) = 4;

validTrials = io.getValidTrials(Exp, 'Grating');
for iTrial = validTrials(:)'
    if ~isfield(Exp.D{iTrial}.PR, 'frozenSequence')
        Exp.D{iTrial}.PR.frozenSequence = false;
    end
end
        
fprintf(fid, 'Done importing session\n');

