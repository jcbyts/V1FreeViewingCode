function [Vm, cBeta, cR, subIdx, cRidge, cLabels, dataIdxs] =  crossValModel(fullR, Vc, cLabels, regIdx, regLabels, folds, dataIdxs)
% function to compute cross-validated rates
%
% modified from https://github.com/churchlandlab/ridgeModel
%
% note: this function has been modified from the version on ChurchlandLab
% github in two ways:
% 1) crossvalidation indices can be passed in and are outputs of the
%    function
% 2) ridgeMML recenter is false meaning an offset parameter is used to
%    capture the baseline rate of the model
% [Vm, cBeta, cR, subIdx, cRidge, cLabels, dataIdxs] =  crossValModel(fullR, Vc, cLabels, regIdx, regLabels, folds, dataIdxs)

cIdx = ismember(regIdx, find(ismember(regLabels,cLabels))); %get index for task regressors
cLabels = regLabels(sort(find(ismember(regLabels,cLabels)))); %make sure motorLabels is in the right order

%create new regressor index that matches motor labels
subIdx = regIdx;
subIdx = subIdx(cIdx);
temp = unique(subIdx);
for x = 1 : length(temp)
    subIdx(subIdx == temp(x)) = x;
end
cR = fullR(:,cIdx);

Vm = zeros(size(Vc),'single'); %pre-allocate motor-reconstructed V

cBeta = cell(1,folds);

% this version of cross validation generates a test set for each fold by
% randomly sampling TIMEPOINTS. This will over-estimate cross-validated r^2
% as long as there are temporal lags in the regressors, which there are.
% This lets the same regressor (e.g., stim onset) from the same trial be in
% both the train and test set. It is better to construct
if ~exist('dataIdxs', 'var')
    warning('crossValModel: using temporal crossvalidation indices')
    rng(1) % for reproducibility
    T = size(Vc,2);
    randIdx = randperm(T); %generate randum number index
    foldCnt = floor(size(Vc,2) / folds);

    dataIdxs = true(folds,size(Vc,2));
    for iFolds = 1:folds
        dataIdxs(iFolds, randIdx(((iFolds - 1)*foldCnt) + (1:foldCnt))) = false; %index for training data
    end
end


for iFolds = 1:folds

    if folds > 1
        dataIdx = dataIdxs(iFolds,:);
        if iFolds == 1
            [cRidge, cBeta{iFolds}] = ridgeMML(Vc(:,dataIdx)', cR(dataIdx,:), false, nan); %get beta weights and ridge penalty for task only model
        else
            [~, cBeta{iFolds}] = ridgeMML(Vc(:,dataIdx)', cR(dataIdx,:), false, cRidge); %get beta weights for task only model. ridge value should be the same as in the first run.
        end

        Vm(:,~dataIdx) = (cR(~dataIdx,:) * cBeta{iFolds}(2:end,:))' + cBeta{iFolds}(1,:)'; %predict remaining data

        if rem(iFolds,folds/5) == 0
            fprintf(1, 'Current fold is %d out of %d\n', iFolds, folds);
        end
    else
        [cRidge, cBeta{iFolds}] = ridgeMML(Vc', cR, true, nan); %get beta weights for task-only model.
        Vm = (cR * cBeta{iFolds})'; %predict remaining data
        disp('Ridgefold is <= 1, fit to complete dataset instead');
    end
end