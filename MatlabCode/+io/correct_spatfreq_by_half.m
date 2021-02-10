function newExp = correct_spatfreq_by_half(Exp)
% Find all protocols with grating objects and divide the SF by 2
% This corrects for a drawing bug in early versions of MarmoV5

% initialize new Exp struct with the original
newExp = Exp;
verbose = true;

%% Find all trials that might have Gratings/Gabors
trialProtocols = cellfun(@(x) x.PR.name, Exp.D, 'uni', 0);

% protocols that might have gratings
protocolsWithGratings = {'ForageProceduralNoise'};

% stimulus classes that have CPD parameters
gratingClasses = {'stimuli.grating', ...
    'stimuli.grating_procedural', ...
    'stimuli.gratingFFnoise', ...
    'stimuli.gabornoise'};

% object properties that refer to spatial frequency
objFields = {'minSF', ...
    'spatialFrequencies', ...
    'cpd', ...
    'cpd2', ...
    'sfRange'};

% trial index
gratingTrials = find(cellfun(@(x) ismember(x, protocolsWithGratings), trialProtocols));
numTrials = numel(gratingTrials);
if verbose
    fprintf('Found %d trials the might use gratings\n', numTrials)
end

for iTrial = 1:numTrials
    
    thisTrial = gratingTrials(iTrial);
    
    % --- Adjust P struct
    fields = {'cpd', 'noncpd'};
    for i = 1:numel(fields)
        field = fields{i};
        
        if isfield(Exp.D{thisTrial}.P, field)
            if verbose
                fprintf('Trial %d) adjusting P struct field: %s\n', thisTrial, field)
            end
            newExp.D{thisTrial}.P.(field) = Exp.D{thisTrial}.P.(field)/2;
            
        end
        
    end
    
    % --- Adjust Gabor / Grating Noise Objects
    if isfield(Exp.D{thisTrial}.PR, 'hNoise')
        hNoise = copy(Exp.D{thisTrial}.PR.hNoise);
        thisType = class(hNoise);
        if ismember(thisType, gratingClasses)
            if verbose
                fprintf('Trial %d) found grating object [%s]\n', thisTrial, thisType)
            end
            
            for i = 1:numel(objFields)
                field = objFields{i};
                if isprop(hNoise, field)
                    if verbose
                        fprintf('\tAdjusting [%s] property\n', field)
                    end
                    hNoise.(field) = hNoise.(field)/2;
                end
            end
        end
       
        newExp.D{thisTrial}.PR.hNoise = copy(hNoise);
        
        % if gratingFF noise then the CPD is saved as a parameter of noise
        % history
        if isa(hNoise, 'stimuli.gratingFFnoise')
            newExp.D{thisTrial}.PR.NoiseHistory(:,3) = Exp.D{thisTrial}.PR.NoiseHistory(:,3)/2;
        end
            
    end
    
    
end