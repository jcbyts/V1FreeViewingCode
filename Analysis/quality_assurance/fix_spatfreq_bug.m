% This should never be run again. was a one-time fix
error("Don't run this file. It was a one-time fix for the already imported files.")
%% Data Meta
dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');

data = readtable(meta_file);

%% Cleanup spatial frequency in MarmoV5 files

marmoV5Sessions = find(strcmp('MarmoV5', data.StimulusSuite));

for ex = marmoV5Sessions(:)'
    
    [Exp, S] = io.dataFactory(ex, 'spike_sorting', []);
    
    fname = fullfile(dataPath, S.processedFileName);
    
    newExp = io.correct_spatfreq_by_half(Exp);
    
    disp('Overwriting File')
    save(fname, '-v7.3', '-struct', 'newExp');
end

%% Fix PLDAPS sessions ori and spatfreq by converting back to kx,ky

pldapsSessions = find(strcmp('PLDAPS', data.StimulusSuite));

for ex = pldapsSessions(:)'
    
    [Exp, S] = io.dataFactory(ex, 'spike_sorting', []);

    fname = fullfile(dataPath, S.processedFileName);

    % Fix kx and ky order in PLDAPS files
    newExp = Exp;

    gratingTrials = io.getValidTrials(Exp, 'Grating');

    for iTrial = 1:numel(gratingTrials)
        thisTrial = gratingTrials(iTrial);
        
        [kx, ky] = pol2cart(Exp.D{thisTrial}.PR.NoiseHistory(:,2)/180*pi, Exp.D{thisTrial}.PR.NoiseHistory(:,3));
        
        [ori, cpd] = cart2pol(ky, kx);
        ori = ori/pi*180;
        ori(ori < 0) = 180 + ori(ori < 0);
        
        newExp.D{thisTrial}.PR.NoiseHistory(:,2) = ori;
        newExp.D{thisTrial}.PR.NoiseHistory(:,3) = cpd;
        
        newExp.D{thisTrial}.PR.spatoris = unique(round(ori, 1));
        newExp.D{thisTrial}.PR.spatfreqs = unique(round(cpd, 1));
    end
    
    disp('Overwriting File')
    save(fname, '-v7.3', '-struct', 'newExp');
    
    
end

