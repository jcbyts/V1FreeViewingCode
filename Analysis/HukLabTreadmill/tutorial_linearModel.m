% Demo code for how to use the linear encoding model used in the study 
% ‘Single-trial neural dynamics are dominated by richly varied movements’ 
% by Musall, Kaufman et al., 2019. 

% apply Musall,Kaufman et al., model to Nienborg data

%% get some data from example recording


%% asign some basic options for the model
% There are three event types when building the design matrix.
% Event type 1 will span the rest of the current trial. Type 2 will span
% frames according to sPostTime. Type 3 will span frames before the event
% according to mPreTime and frames after the event according to mPostTime.
opts.frameRate = D.frameRate;
blocks = D.blocks;
frames = blocks(:,2) - blocks(:,1);
assert(all(frames==unique(frames)), 'makeDesignMatrix: there are more than one trial length. this is a problem')
opts.frames = unique(frames);
opts.sPostTime = ceil(.1 * opts.frameRate);   % follow stim events for sPostStim in frames (used for eventType 2)
opts.mPreTime = ceil(0.5 * opts.frameRate);  % precede motor events to capture preparatory activity in frames (used for eventType 3)
opts.mPostTime = ceil(2 * opts.frameRate);   % follow motor events for mPostStim in frames (used for eventType 3)
opts.folds = 10; %nr of folds for cross-validation

%% Task Events

% task events
hdx_left_labels = arrayfun(@(x) sprintf('hdx_left_%02.2f', x), D.hdx_levels, 'uni', 0);
hdx_right_labels = arrayfun(@(x) sprintf('hdx_right_%02.2f', x), D.hdx_levels, 'uni', 0);
nhdx = numel(D.hdx_levels);
clear taskEvents

ctr = 1;
taskLabels = [hdx_left_labels, hdx_right_labels, {'contrast_left', 'contrast_right'}];
taskEventType = [2*ones(1, nhdx) 2*ones(1, nhdx) 1 1]; %different type of events.

taskEvents(:,ctr:nhdx) = D.hdx_left;
covariateStarts = ctr;

ctr = ctr + nhdx;
taskEvents(:,ctr:(ctr+nhdx-1)) = D.hdx_right;
covariateStarts = [covariateStarts ctr];

ctr = ctr + nhdx;
taskEvents(:,ctr) = [0; diff(D.contrast_left)==1];
covariateStarts = [covariateStarts ctr];

ctr = ctr + 1;
taskEvents(:,ctr) = [0; diff(D.contrast_right)==1];
covariateStarts = [covariateStarts ctr];

%% movement events
moveLabels = {'response' 'rGrab' 'lLick' 'rLick' 'nose' 'whisk'}; %some movement variables
% moveEventType = [3 3 3 3 3 3]; %different type of events. these are all peri-event variables.
% for x = 1 : length(moveLabels)
%     moveEvents(:,x) = fullR(:,find(recIdx == find(ismember(recLabels, moveLabels(x))),1)+15); %find movement regressor.
% end
% clear fullR %clear old design matrix

%%
[taskR, taskIdx] = makeDesignMatrix(taskEvents, taskEventType, opts); %make design matrix for task variables



% 
% % make design matrix
% [taskR, taskIdx] = makeDesignMatrix(taskEvents, taskEventType, opts); %make design matrix for task variables
% [moveR, moveIdx] = makeDesignMatrix(moveEvents, moveEventType, opts); %make design matrix for movement variables
% 
% fullR = [taskR, moveR, vidR]; %make new, single design matrix
% moveLabels = [moveLabels, {'video'}];
% regIdx = [taskIdx; moveIdx + max(taskIdx); repmat(max(moveIdx)+max(taskIdx)+1, size(vidR,2), 1)]; %regressor index
% regLabels = [taskLabels, moveLabels];


vidR = [D.faceSVD D.bodySVD];
fullR = [double(taskR) zscore(double(vidR))];
regIdx = [taskIdx; repmat(max(taskIdx)+1, size(vidR,2), 1)];
regLabels = [taskLabels, 'video'];
%% run QR and check for rank-defficiency. This will show whether a given regressor is highly collinear with other regressors in the design matrix.
% The resulting plot ranges from 0 to 1 for each regressor, with 1 being
% fully orthogonal to all preceeding regressors in the matrix and 0 being
% fully redundant. Having fully redundant regressors in the matrix will
% break the model, so in this example those regressors are removed. In
% practice, you should understand where the redundancy is coming from and
% change your model design to avoid it in the first place!

rejIdx = false(1,size(fullR,2));
[~, fullQRR] = qr(bsxfun(@rdivide,fullR,sqrt(sum(fullR.^2))),0); %orthogonalize normalized design matrix
figure; plot(abs(diag(fullQRR)),'linewidth',2); ylim([0 1.1]); title('Regressor orthogonality'); drawnow; %this shows how orthogonal individual regressors are to the rest of the matrix
axis square; ylabel('Norm. vector angle'); xlabel('Regressors');
if sum(abs(diag(fullQRR)) > max(size(fullR)) * eps(fullQRR(1))) < size(fullR,2) %check if design matrix is full rank
    temp = ~(abs(diag(fullQRR)) > max(size(fullR)) * eps(fullQRR(1)));
    fprintf('Design matrix is rank-defficient. Removing %d/%d additional regressors.\n', sum(temp), sum(~rejIdx));
    rejIdx(~rejIdx) = temp; %reject regressors that cause rank-defficint matrix
end
% save([fPath filesep 'regData.mat'], 'fullR', 'regIdx', 'regLabels','fullQRR','-v7.3'); %save some model variables

%% fit model to spike data
Vc = double(D.robs');

[ridgeVals, dimBeta] = ridgeMML(Vc', fullR, true); %get ridge penalties and beta weights.
% save([fPath 'dimBeta.mat'], 'dimBeta', 'ridgeVals'); %save beta kernels

%% mask by probe
probes = unique({D.units.probe});
U = [];
for i = 1:numel(probes)
    U = [U; ismember({D.units.probe}, probes{i})];
end

% treat each neuron individually
U = eye(size(D.robs,2));

%%
% (X'X + lambda*eye(size(X,2)))\(X'*Y)

%reconstruct data and compute R^2
Vm = (fullR * dimBeta)';
corrMat = modelCorr(Vc,Vm,U) .^2; %compute explained variance

figure(1); clf
bar(corrMat)
ylabel('Var Explained')
xlabel('Neuron')

%% check beta kernels
% select variable of interest. Must be included in 'regLabels'.

% find beta weights for current variable
cvars = find(contains(regLabels, 'hdx'));
nvars = numel(cvars);

nlags = opts.sPostTime+1;
betas = zeros(nvars, nlags, size(D.robs,2));


for i = 1:nvars
    cIdx = regIdx==cvars(i);
    betas(i,:,:) = dimBeta(cIdx,:);
end

%%
cc = cc + 1;
figure(1); clf
hdx_tuning = squeeze(betas(:,:,cc));
xx = repmat(1:nlags, nvars, 1)';
yy = repmat(1:nvars, nlags, 1);
zz = hdx_tuning';
h = plot3(xx, yy, zz , 'k'); hold on
j = plot3(xx(:,1:nvars/2), yy(:,1:nvars/2), zz(:,1:nvars/2) , 'b'); hold on
legend([h(1) j(1)], {'Left Stim', 'Right Stim'})
set(gca, 'YTick', 1:nvars, 'YTickLabel', strrep(regLabels(cvars), '_', ' '), 'YTickLabelRotation', 45)
ylabel('Disparity')
xlabel('Time lags')
title([D.units(cc).probe ' ' D.units(cc).electrode])

%%
U = reshape(U, [], size(Vc,1)); 
cBeta = U * dimBeta(cIdx, :)';
figure(1); clf
imagesc(cBeta)

%% run cross-validation
%full model - this will take a moment
[Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR, Vc, regLabels, regIdx, regLabels, opts.folds);
save(fullfile(pwd, 'cvFull.mat'), 'Vfull', 'fullBeta', 'fullR', 'fullIdx', 'fullRidge', 'fullLabels'); %save some results

%%
fullMat = modelCorr(Vc,Vfull,U) .^2; %compute explained variance

%task model alone - this will take a moment
[Vtask, taskBeta, taskR, taskIdx, taskRidge, taskLabels] = crossValModel(fullR, Vc, taskLabels, regIdx, regLabels, opts.folds);
save(fullfile(pwd, 'cvTask.mat'), 'Vtask', 'taskBeta', 'taskR', 'taskIdx', 'taskRidge', 'taskLabels'); %save some results

taskMat = modelCorr(Vc,Vtask,U) .^2; %compute explained variance


%%


figure(1); clf
set(gcf, 'Color', 'w')
plot(fullMat, taskMat, 'o')
hold on
plot(xlim, xlim, 'k')
xlabel('Variance Explained (Full)')
ylabel('Variance Explained (Task Only)')

%%

%movement model alone - this will take a moment
[Vmove, moveBeta, moveR, moveIdx, moveRidge, moveLabels] = crossValModel(fullR, Vc, moveLabels, regIdx, regLabels, opts.folds);
save(fullfile(pwd, 'cvMove.mat'), 'Vmove', 'moveBeta', 'moveR', 'moveIdx', 'moveRidge', 'moveLabels'); %save some results

moveMat = modelCorr(Vc,Vmove,U) .^2; %compute explained variance
moveMat = arrayShrink(moveMat,mask,'split'); %recreate move frame
moveMat = alignAllenTransIm(moveMat,opts.transParams); %align to allen atlas
moveMat = moveMat(:, 1:size(allenMask,2));

%% show R^2 results
%cross-validated R^2
figure;
subplot(1,3,1);
mapImg = imshow(fullMat,[0 0.75]);
colormap(mapImg.Parent,'inferno'); axis image; title('cVR^2 - Full model');
set(mapImg,'AlphaData',~isnan(mapImg.CData)); %make NaNs transparent.
        
subplot(1,3,2);
mapImg = imshow(taskMat,[0 0.75]);
colormap(mapImg.Parent,'inferno'); axis image; title('cVR^2 - Task model');
set(mapImg,'AlphaData',~isnan(mapImg.CData)); %make NaNs transparent.
  
subplot(1,3,3);
mapImg = imshow(moveMat,[0 0.75]);
colormap(mapImg.Parent,'inferno'); axis image; title('cVR^2 - Movement model');
set(mapImg,'AlphaData',~isnan(mapImg.CData)); %make NaNs transparent.
  
%unique R^2
figure;
subplot(1,2,1);
mapImg = imshow(fullMat - moveMat,[0 0.4]);
colormap(mapImg.Parent,'inferno'); axis image; title('deltaR^2 - Task model');
set(mapImg,'AlphaData',~isnan(mapImg.CData)); %make NaNs transparent.

subplot(1,2,2);
mapImg = imshow(fullMat - taskMat,[0 0.4]);
colormap(mapImg.Parent,'inferno'); axis image; title('deltaR^2 - Movement model');
set(mapImg,'AlphaData',~isnan(mapImg.CData)); %make NaNs transparent.

