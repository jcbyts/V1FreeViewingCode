%% MAIN IMPORT HELPER SCRIPT
% The way you import a session (assuming it has already been imported using
% the standard marmopipe pre-processing) is to simply call the dataFactory
% from a computer that is connected to the lab server.
% If the session has already been imported, it will be loaded. Any missing
% pieces will be found on the server.


%% List all sessions that have been entered in the dataset spreadsheet
sesslist = io.dataFactory();

%%
dataPath = fullfile(fileparts(which('addFreeViewingPaths')), 'Data');
fname = fullfile(dataPath, 'import_log.txt');
fid = fopen(fname, 'w+');

%%
fseek(fid, 0, 1);

if ~exist(fname, 'file')
    fprintf(fid, 'IMPORT LOG %s\n', datestr(now));
end





%% Import session by number or by name
for iEx = 61:98 
fprintf(fid, '**********************************\n**********************************\n\n');
% main import. pass 4 arguments out to import LFP and MUA as well
c = evalc("[Exp, S, lfp, mua] = io.dataFactory(iEx, 'spike_sorting', 'kilo');");
fprintf(fid, c);

% we use spike waveforms for some neuron classification so clip them out
sp = io.clip_waveforms(Exp, S);
fprintf(fid, 'Clipping Spike Waveforms %s\n', datestr(now));
disp('Done')

fprintf(fid, 'SUCCESS\n**********************************\n**********************************\n\n');
end