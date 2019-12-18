function validTrials = getValidTrials(Exp, stimulusSet)
% validTrials = getValidTrials(Exp, stimulusSet)

trialProtocols = cellfun(@(x) x.PR.name, Exp.D, 'uni', 0);

% --- find the trials that we want to analyze
ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS) | ~isnan(x.END_EPHYS), Exp.D));
if (numel(ephysTrials)/numel(Exp.D)) < 0.6
    disp('Something is wrong. Assuming all trials had ephys')
    ephysTrials = 1:numel(Exp.D);
end


switch stimulusSet
    case {'Grating'}
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==1, Exp.D(validTrials)));
        
    case {'Gabor'}
        % Forage trials
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==4, Exp.D(validTrials)));
        
    case {'Dots'}
        % Forage trials
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        % dot spatial noise trials
        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));

        dotSize = cellfun(@(x) x.P.dotSize, Exp.D(validTrials));
        validTrials = validTrials(dotSize==min(dotSize));
    case {'BackImage'}
        validTrials = intersect(find(strcmp(trialProtocols, 'BackImage')), ephysTrials);
        
    case {'Forage'}
        
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);
    
    case {'FixFlash'}
        
        validTrials = find(strcmp(trialProtocols, 'FixFlash'));
        
    case {'FaceCal'}
        
        validTrials = find(strcmp(trialProtocols, 'FaceCal'));
        
    otherwise
        % use all valid conditions (BackImage or
        validTrials = find(strcmp(trialProtocols, 'BackImage') | strcmp(trialProtocols, 'ForageProceduralNoise'));
        validTrials = intersect(validTrials, ephysTrials);
end


numValidTrials = numel(validTrials);
fprintf('Found %d valid trials\n', numValidTrials)