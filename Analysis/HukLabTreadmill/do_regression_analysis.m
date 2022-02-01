function [Stim, opts, Rpred, Running] = do_regression_analysis(D)
% [Stim, opts, Rpred, Running] = do_regression_analysis(D)

%% Get all relevant covariates
Stim = [];
opts = [];
Rpred = [];
Running = [];

Stim = convert_Dstruct_to_trials(D, 'bin_size', 1/60, 'pre_stim', .2, 'post_stim', .2);

%% bin spikes
if isempty(Stim)
    return
end

[Robs, cids] = bin_spikes_at_frames(Stim, D);

Robs = filtfilt(ones(7,1)/7, 1, Robs);

NC = size(Robs,2);
U = eye(NC);
cc = 0;

%%
Rt = reshape(Robs, numel(Stim.trial_time), [], NC);
figure(1); clf;
plot(squeeze(sum(sum(Rt,2),1)), sum(Robs), '.'); hold on; plot(xlim, xlim, 'k')

Rt = squeeze(sum(Rt));
size(Rt)
% 
% assert(sum(Robs)==sum(Rt))
%% get data filters
dfs = false(numel(Stim.trial_time), numel(Stim.grating_onsets), NC);

for cc = 1:NC
iix = getStableRange(imboxfilt(Rt(:,cc), 25));
dfs(:,iix,cc) = true;
%     dfs(iix,cc) = true;
drawnow
end

dfs = reshape(dfs, [], NC);
% figure(2); clf
% plot(zscore(Rt(:,cc)))
% end

%% 
[~, ind] = sort(var(Rt)./sum(Rt), 'descend');
figure(1); clf
imagesc(filtfilt(ones(100,1)/100, 1, Robs(:,ind)))

figure(2); clf
plot(var(Rt)/numel(Stim.trial_time), var(Robs), '.'); hold on
plot(xlim, xlim, 'k')

%% get unit eccentricity
ecc = nan(1, NC);
if isfield(D, 'units')
    allcids = unique(D.spikeIds);
    rfinds = (arrayfun(@(x) x.maxV > 10 & x.area > .5 , D.units));
    rfcids = allcids(rfinds);
    eccrf = arrayfun(@(x) hypot(x.center(1), x.center(2)), D.units(rfinds));
    ecc(ismember(cids, rfcids)) = eccrf(ismember(rfcids, cids));
end

%% build the design matrix components
% build design matrix
opts = struct();
assert(Stim.bin_size==1/60, 'Parameters have only been set for 60Hz sampling')
opts.collapse_speed = true;
opts.collapse_phase = true;
opts.include_onset = false;
opts.stim_dur = median(D.GratingOffsets-D.GratingOnsets) + 0.2; % length of stimulus kernel
opts.use_sf_tents = false;

stim_dur = ceil((opts.stim_dur)/Stim.bin_size);
opts.stim_ctrs = [1:2:10 15:15:stim_dur];
if ~isfield(opts, 'stim_ctrs')
    opts.stim_ctrs = [0:5:10 15:10:stim_dur-2];
end
Bt = tent_basis(0:stim_dur+15, opts.stim_ctrs);

figure(1); clf
plot(Bt)

%%
[X, opts] = build_design_matrix(Stim, opts);


% concatenate full design matrix


label = 'Stim';
regLabels = {label};
k = 1;
X_ = X{ismember(opts.Labels, label)};
regIdx = repmat(k, 1, size(X_,2)); %#ok<REPMAT>
fullR = X_;

if opts.include_onset
    label = 'Stim Onset';
    regLabels = [regLabels {label}];
    k = k + 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = [regIdx repmat(k, [1, size(X_,2)])];
    fullR = [fullR X_];
end

% add additional covariates
label = 'Drift';
regLabels = [regLabels {label}];
k = k + 1;
X_ = X{ismember(opts.Labels, label)};
regIdx = [regIdx repmat(k, [1, size(X_,2)])];
fullR = [fullR X_];

label = 'Saccade';
regLabels = [regLabels {label}];
k = k + 1;
X_ = X{ismember(opts.Labels, label)};
regIdx = [regIdx repmat(k, [1, size(X_,2)])];
fullR = [fullR X_];

label = 'Running';
regLabels = [regLabels {label}];
k = k + 1;
X_ = X{ismember(opts.Labels, label)};
regIdx = [regIdx repmat(k, [1, size(X_,2)])];
fullR = [fullR X_];

isrunning = Stim.tread_speed(:) > opts.run_thresh;
zpupil = Stim.eye_pupil(:) / nanstd(Stim.eye_pupil(:)); %#ok<*NANSTD> 
zpupil = zpupil - nanmean(zpupil); %#ok<*NANMEAN> 
ispupil = zpupil > opts.pupil_thresh;

regLabels = [regLabels {'Is Running'}];
k = k + 1;
regIdx = [regIdx k];
fullR = [fullR isrunning(:)];

regLabels = [regLabels {'Is Pupil'}];
k = k + 1;
regIdx = [regIdx k];
fullR = [fullR ispupil(:)];

assert(size(fullR,2) == numel(regIdx), 'Number of covariates does not match Idx')
assert(numel(unique(regIdx)) == numel(regLabels), 'Number of labels does not match')

for i = 1:numel(regLabels)
    label= regLabels{i};
    cov_idx = regIdx == find(strcmp(regLabels, label));
    fprintf('[%s] has %d parameters\n', label, sum(cov_idx))
end
% 
% opts.design_remove_cols = find(sum(fullR)<2);
% opts.design_remove_idx = regIdx(opts.design_remove_cols);
% fullR(:,opts.design_remove_cols) = [];
% regIdx(opts.design_remove_cols) = [];
% fprintf('Removed %d columns with no regressors\n', numel(opts.design_remove_cols))

% %% run QR and check for rank-defficiency. This will show whether a given regressor is highly collinear with other regressors in the design matrix.
% % The resulting plot ranges from 0 to 1 for each regressor, with 1 being
% % fully orthogonal to all preceeding regressors in the matrix and 0 being
% % fully redundant. Having fully redundant regressors in the matrix will
% % break the model, so in this example those regressors are removed. In
% % practice, you should understand where the redundancy is coming from and
% % change your model design to avoid it in the first place!
% reg_inds = find(regIdx > 0);
% rejIdx = false(1,size(fullR(good_inds,reg_inds),2));
% [~, fullQRR] = qr(bsxfun(@rdivide,fullR(good_inds,reg_inds),sqrt(sum(fullR(good_inds,reg_inds).^2))),0); %orthogonalize normalized design matrix
% figure; plot(abs(diag(fullQRR)),'linewidth',2); ylim([0 1.1]); title('Regressor orthogonality'); drawnow; %this shows how orthogonal individual regressors are to the rest of the matrix
% axis square; ylabel('Norm. vector angle'); xlabel('Regressors');
% if sum(abs(diag(fullQRR)) > max(size(fullR(good_inds,reg_inds))) * eps(fullQRR(1))) < size(fullR(good_inds,reg_inds),2) %check if design matrix is full rank
%     temp = ~(abs(diag(fullQRR)) > max(size(fullR(good_inds,reg_inds))) * eps(fullQRR(1)));
%     fprintf('Design matrix is rank-defficient. Removing %d/%d additional regressors.\n', sum(temp), sum(~rejIdx));
%     rejIdx(~rejIdx) = temp; %reject regressors that cause rank-defficint matrix
% end
% % save([fPath filesep 'regData.mat'], 'fullR', 'regIdx', 'regLabels','fullQRR','-v7.3'); %save some model variables
   

%% Build cv indices that are trial-based
good_inds = find(~(isnan(Stim.eye_pos_proj_adjusted(:)) | isnan(Stim.tread_speed(:))));

folds = 5;
n = numel(Stim.trial_time);
T = size(Robs,1);

dataIdxs = true(folds, T);
rng(1)
trorder = randperm(T / n);
for t = 1:(T/n)
    i = mod(t, folds);
    if i == 0, i = folds; end
    dataIdxs(i,(trorder(t)-1)*n + (1:n)) = false;
end


dataIdxs = dataIdxs(:,good_inds);
    

   
%% Fit full model


% try different models
modelNames = {'nostim', 'stim', 'stimsac', 'stimrun', 'stimrunsac', 'drift'};

excludeParams = { {'Stim', 'Stim Onset'}, {'Running', 'Saccade'}, {'Running'}, {'Saccade'}, {}, {'Stim','Stim Onset','Running','Saccade'} };
alwaysExclude = {'Stim R', 'Stim S', 'Is Running', 'Is Pupil'};

Rpred = struct();

models2fit = {'stim', 'drift'};

for iModel = find(ismember(modelNames, models2fit))
    fprintf('Fitting Model [%s]\n', modelNames{iModel})
    exclude = [excludeParams{iModel} alwaysExclude];
    modelLabels = setdiff(regLabels, exclude); %#ok<*NASGU>
%     evalc("[Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR(good_inds,:), Robs(good_inds,:)', modelLabels, regIdx, regLabels, folds, dataIdxs);");
    [Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel((fullR(good_inds,:)), Robs(good_inds,:)', modelLabels, regIdx, regLabels, folds, dataIdxs, dfs(good_inds,:));
    Rpred.(modelNames{iModel}).Rpred = Vfull;
    Rpred.(modelNames{iModel}).Offset = cell2mat(cellfun(@(x) x(1,:), fullBeta(:), 'uni', 0));
    for i = 1:numel(fullBeta)
        fullBeta{i}(1,:) = [];
    end
    Rpred.(modelNames{iModel}).Beta = fullBeta; %#ok<*NODEF>
    Rpred.(modelNames{iModel}).covIdx = fullIdx;
    Rpred.(modelNames{iModel}).Ridge = fullRidge;
    Rpred.(modelNames{iModel}).Labels = fullLabels;
    
    rbar = nanmean(Robs(good_inds,:));
    Rpred.(modelNames{iModel}).Rsquared = rsquared(Robs(good_inds,:), Vfull', false, rbar); %compute explained variance
    Rpred.(modelNames{iModel}).CC = modelCorr(Robs(good_inds,:)',Vfull,eye(NC)); %compute explained variance  
end

fprintf('Done\n')

%%


%%

[~, ind] = sort(Rpred.drift.Rsquared, 'descend');
figure(1); clf
fields = models2fit;
for i = 1:numel(fields)
    plot(Rpred.(fields{i}).Rsquared(ind), '.'); hold on
end
plot(xlim, [0 0], 'k')

ii = 1;
%% test gain drift model

restLabels = [{'Stim Onset'}    {'Drift'}];
% get drift rate
drift_model = 'drift';
model = 'driftgain';
ii = ii + 1;
cc = ind(ii);
% X = [fullR(good_inds,:) 


Rpred.(model).Gains = nan(folds, 2, NC);
Rpred.(model).Rpred = nan(NC, NT);
Rpred.(model).stimgain = nan(NC, NT, folds);
Rpred.(model).Beta = cell(folds,1);

fprintf('%d/%d Unit\n', cc, NC)
Lgain = nan;
Lfull = nan;
covLabels = [regLabels {'Drift Output'}];
covIdx = [regIdx max(regIdx) + 1];

for ifold = 1:folds
    train_inds = find(dataIdxs(ifold,:))';
    test_inds = find(~dataIdxs(ifold,:))';
    
    % get drift for this cell
    driftrate = fullR(good_inds, ismember(regIdx, find(ismember(regLabels, 'Drift'))))*Rpred.stim.Beta{ifold}(ismember(Rpred.stim.covIdx, find(ismember(Rpred.stim.Labels, 'Drift'))),cc);
    driftrate = driftrate + min(driftrate)+1;
%     driftrate = Rpred.stim.Offset(ifold,cc) + driftrate;
%     driftrate = Rpred.drift.Rpred(cc,:)';
    X = [fullR(good_inds,:) driftrate];
    
    %             evalc("[Betas, Gain, Ridge, Rhat, Lgain, Lfull] = AltLeastSqGainModel(fullR(good_inds,:), Robs(good_inds,cc), train_inds, regIdx, regLabels, {'Stim'}, GainTerm(iModel), restLabels, Lgain, Lfull);");
    [Betas, Gain, Ridge, Rhat, Lgain, Lfull] = AltLeastSqGainModel(X, Robs(good_inds,cc), train_inds, ...
        covIdx, covLabels, {'Stim'}, 'Drift Output', restLabels, Lgain, Lfull);
    Rpred.(model).stimgain(cc,:,ifold) = Gain(1) + driftrate*Gain(2);
    Rpred.(model).Offset(ifold,cc) = Betas(1);
    Rpred.(model).Beta{ifold}(:,cc) = Betas(2:end);
    Rpred.(model).Gains(ifold,:,cc) = Gain;
    Rpred.(model).Ridge(ifold,cc) = Ridge;
    Rpred.(model).Rpred(cc,test_inds) = Rhat(test_inds);
end

r2 = rsquared(Robs(good_inds,cc), Rpred.(model).Rpred(cc,:)');
fprintf('r2: %02.3f (stim), %02.3f (drift), %02.3f (drift gain)\n', Rpred.stim.Rsquared(cc), Rpred.drift.Rsquared(cc), r2)

stim_on = Stim.contrast(:) > 0;



figure(1); clf
plot(Robs(good_inds,cc), 'k'); hold on
plot(Rpred.(model).Rpred(cc,:))
plot(squeeze(Rpred.(model).stimgain(cc,:,:)), 'r')
g = mean(squeeze(Rpred.(model).stimgain(cc,:,:)),2);
d = g > .5*prctile(g, 75);
plot(d, 'Linewidth', 2)
% plot()
% plot(g)
% 
% figure(2); clf
% plot(Rpred.(model).Gains(:,:,cc), 'Linewidth', 2)
% plot(Rhat)


%%
cc = cc + 1;
if cc > NC
    cc = 1;
end
X = fullR(good_inds,:);
Y = Robs(good_inds,cc);
NT = numel(good_inds);
nsteps = 5;

figure(1); clf
r2 = nan(nsteps,nsteps);
rhbar = nan(nsteps,1);
for i = 1:nsteps
    iix = (i-1)*ceil(NT/nsteps) + (1:ceil(NT/nsteps));
    iix(iix > NT) = [];
%     ix = ~ismember(regIdx, find(ismember(regLabels, 'Drift')));
    ix = true(1,size(X,2));
    [~, betas, ~] = ridgeMML(Y(iix), X(iix,ix), false);

    for j = 1:nsteps
        
        iix = (j-1)*ceil(NT/nsteps) + (1:ceil(NT/nsteps));
        iix(iix > NT) = [];

        Rhat = betas(1) + X(iix,ix)*betas(2:end);
    
        r2(i,j) = rsquared(Y(iix), Rhat);
%         rhbar(i) = var(Rhat/Stim.bin_size);
%         subplot(2,1,1)
%         plot(iix, Rhat); hold on
%         plot(iix, ones(numel(iix),1)*r2(i), 'k')
    end
end

figure(1); clf
imagesc(max(r2,0));
title(cc)
colormap parula
colorbar
%%

rhbar = rhbar./max(rhbar);

subplot(2,1,2)
plot(r2); hold on
plot(rhbar)
plot(r2.*rhbar, 'k')
plot(xlim, .2*[1 1], 'k--')



%%

figure(1); clf
plot(Robs(good_inds,cc), 'k'); hold on
plot(Rpred.(model).Rpred(cc,:))
hold on
plot(Rhat)


Rtrue = Robs(good_inds,cc);
% Rhat = Rpred.(model).Rpred(cc,:)';
Rbar = ones(numel(good_inds,1))*mean(Rtrue);
Rbar = filtfilt(ones(5000,1)/5000, 1, Rtrue);

resid = (Rtrue- Rhat).^2;
rtot = (Rtrue - Rbar).^2;

bcar = ones(500,1)/500;
resid = filtfilt(bcar, 1, resid);
rtot = filtfilt(bcar, 1, rtot);

figure(2); clf

plot(rtot); hold on
plot(resid)
plot(1-resid./rtot)
ylim([-1 1])
% plot(stim_on(good_inds))
% plot(zpupil(good_inds))

%%
%% Get PSTH
model = 'driftgain';
model2 = 'stim';
Rhat = Rpred.(model).Rpred';
Rhat2 = Rpred.(model2).Rpred';

nbins = numel(Stim.trial_time);
B = eye(nbins);

dirs = max(filter([1; -1], 1, Stim.direction(:)==opts.directions(:)'), 0);
% dirs = double([zeros(1,opts.nd); diff(Stim.direction(:)==opts.directions(:)')==1]);
tax = Stim.trial_time;

n = sum(tax<=0);
Xt = temporalBases_dense(circshift(dirs, -n), B);


XY = (Xt(good_inds,:)'*Robs(good_inds,:)) ./ sum(Xt(good_inds,:))';
XYhat = (Xt(good_inds,:)'*Rhat) ./ sum(Xt(good_inds,:))';
XYhat2 = (Xt(good_inds,:)'*Rhat2) ./ sum(Xt(good_inds,:))';

tax = Stim.trial_time;

Xbar = reshape(XYhat, [nbins, opts.nd, NC]);
Xbar2 = reshape(XYhat2, [nbins, opts.nd, NC]);
Rbar = reshape(XY, [nbins, opts.nd, NC]);

figure(1); clf
subplot(1,2,1)
imagesc(tax, opts.directions, Xbar(:,:,cc)')
subplot(1,2,2)
imagesc(tax, opts.directions, Rbar(:,:,cc)')

figure(2); clf
subplot(1,2,2)
plot(tax, Rbar(:,:,cc))
yd = ylim;
xlim(tax([1 end]))


rbar = nanmean(reshape(Rbar(:,:,cc), [], 1));
r2 = rsquared(Rbar(:,:,cc), Xbar(:,:,cc), true, rbar);
r2m2 = rsquared(Rbar(:,:,cc), Xbar2(:,:,cc), true, rbar);

title(sprintf('%02.2f, %02.2f', r2, r2m2))

subplot(1,2,1)
plot(tax, Xbar(:,:,cc))
ylim(yd)
xlim(tax([1 end]))

title(sprintf('%02.2f, %02.2f', Rpred.(model).Rsquared(cc), Rpred.(model2).Rsquared(cc)))



%%
figure(1); clf, 
imagesc(Robs(good_inds,ind)')
colormap(1-gray)
hold on
tread_speed = Stim.tread_speed(:);
t = imgaussfilt(tread_speed(good_inds)*100, 50);
plot(t, 'r')


%% Fit Gain models for running and pupil
restLabels = [{'Stim Onset'}    {'Drift'}    {'Saccade'}    {'Running'}];
GainModelNames = {'RunningGain', 'PupilGain'}; %{'nostim', 'stim', 'stimsac', 'stimrun', 'stimrunsac'};
GainTerm = {'Is Running', 'Is Pupil'};
for iModel = 1%:numel(GainTerm)
    
    labelIdx = ismember(regLabels, [{'Stim'} restLabels]);
    covIdx = regIdx(ismember(regIdx, find(labelIdx)));
    covLabels = regLabels(labelIdx);
    [~, ~, covIdx] = unique(covIdx);
    
    Rpred.(GainModelNames{iModel}).covIdx = covIdx;
    Rpred.(GainModelNames{iModel}).Beta = cell(folds,1);
    Rpred.(GainModelNames{iModel}).Offset = zeros(folds, NC);
    Rpred.(GainModelNames{iModel}).Gains = zeros(folds, 2, NC);
    Rpred.(GainModelNames{iModel}).Labels = covLabels;
    Rpred.(GainModelNames{iModel}).Rpred = zeros(NC, numel(good_inds));
    Rpred.(GainModelNames{iModel}).Ridge = zeros(folds, NC);
    
    ndim = numel(covIdx);
    for ifold = 1:folds
        Rpred.(GainModelNames{iModel}).Beta{ifold} = zeros(ndim, NC);
    end
    
    
    for cc = 1:NC
        fprintf('%d/%d Unit\n', cc, NC)
        Lgain = nan;
        Lfull = nan;

        for ifold = 1:folds
            train_inds = find(dataIdxs(ifold,:))';
            test_inds = find(~dataIdxs(ifold,:))';

%             evalc("[Betas, Gain, Ridge, Rhat, Lgain, Lfull] = AltLeastSqGainModel(fullR(good_inds,:), Robs(good_inds,cc), train_inds, regIdx, regLabels, {'Stim'}, GainTerm(iModel), restLabels, Lgain, Lfull);");
            [Betas, Gain, Ridge, Rhat, Lgain, Lfull] = AltLeastSqGainModel(fullR(good_inds,:), Robs(good_inds,cc), train_inds, regIdx, regLabels, {'Stim'}, GainTerm(iModel), restLabels, Lgain, Lfull);
            Rpred.(GainModelNames{iModel}).Offset(ifold,cc) = Betas(1);
            Rpred.(GainModelNames{iModel}).Beta{ifold}(:,cc) = Betas(2:end);
            Rpred.(GainModelNames{iModel}).Gains(ifold,:,cc) = Gain;
            Rpred.(GainModelNames{iModel}).Ridge(ifold,cc) = Ridge;
            Rpred.(GainModelNames{iModel}).Rpred(cc,test_inds) = Rhat(test_inds);
        end
        
    end
    
    % evaluate model
    rbar = nanmean(Robs(good_inds,:));
    Rhat = Rpred.(GainModelNames{iModel}).Rpred';
    Rpred.(GainModelNames{iModel}).Rsquared = rsquared(Robs(good_inds,:), Rhat, false, rbar); %compute explained variance
    Rpred.(GainModelNames{iModel}).CC = modelCorr(Robs(good_inds,:)',Rhat',U); %compute explained variance
    fprintf('Done\n')
    
end
%% Get stimulus prediction to create running-dependent regressor
model = 'stimsac';

NT = numel(good_inds);
assert(NT == size(dataIdxs,2), 'Number of samples does not match')

stimrateTest = zeros(NT, NC);
stimrateTrain = zeros(NT, NC);
allrate = zeros(NT,NC);
driftrate = zeros(NT, NC);
r0 = ~dataIdxs'*Rpred.(model).Offset;
for ifold = 1:folds
    
    test_inds = (~dataIdxs(ifold,:));
    
    
    Xtest = fullR(good_inds(test_inds),:);
    
    % stim only prediction
    idx = ismember(Rpred.(model).covIdx, find(ismember(Rpred.(model).Labels, {'Stim'})));
    stimpred = Xtest(:,idx) * Rpred.(model).Beta{ifold}(idx,:);
    stimrateTest(test_inds,:) = stimpred;
    
    % stim + drift + stim onset
    idx = ismember(Rpred.(model).covIdx, find(ismember(Rpred.(model).Labels, {'Stim', 'Stim Onset', 'Drift', 'Saccade'})));
    allpred = Xtest(:,idx) * Rpred.(model).Beta{ifold}(idx,:);
    allrate(test_inds,:) = allpred;
    
    % drift only
    idx = ismember(Rpred.(model).covIdx, find(ismember(Rpred.(model).Labels, {'Drift'})));
    driftpred = Xtest(:,idx) * Rpred.(model).Beta{ifold}(idx,:);
    driftrate(test_inds,:) = driftpred;
end

driftrate = driftrate + r0;
%% Show model comparison
models = fieldnames(Rpred);
nModels = numel(models);
model_pairs = nchoosek(1:nModels, 2);
npairs = size(model_pairs,1);

% first get the xlims for the comparison
m = 0;
for i = 1:npairs
    m = max(max(abs(Rpred.(models{model_pairs(i,1)}).Rsquared- Rpred.(models{model_pairs(i,2)}).Rsquared)),0);
end

inc_thresh = .01;
figure(1); clf
sx = ceil(sqrt(npairs));
sy = round(sqrt(npairs));

for i = 1:npairs
    subplot(sx, sy, i)
    
    m1 = Rpred.(models{model_pairs(i,1)}).Rsquared;
    m1name = models{model_pairs(i,1)};
    m2 = Rpred.(models{model_pairs(i,2)}).Rsquared;
    m2name = models{model_pairs(i,2)};
    
    if mean(m1) > mean(m2)
        mtmp = m1;
        mtmpname = m1name;
        m1 = m2;
        m1name = m2name;
        m2 = mtmp;
        m2name = mtmpname;
    end
    
    plot(m1, m2, 'ow', 'MarkerSize', 4, 'MarkerFaceColor', repmat(.5, 1, 3)); hold on
    plot(xlim, xlim, 'k')
    plot([0 inc_thresh], [inc_thresh inc_thresh], 'k--')
    plot([inc_thresh inc_thresh], [0 inc_thresh], 'k--')
    xlabel(m1name)
    ylabel(m2name)
    xlim(max(xlim, 0))
    ylim(max(ylim, 0))
    
%     subplot(2,npairs,i+npairs)
%     iix = m2 > inc_thresh;
%     histogram(m2(iix) - m1(iix), ceil(sum(iix)/2), 'FaceColor', repmat(.5, 1, 3))
%     xlabel(sprintf('%s - %s', m2name, m1name))
%     xlim([-m*.5 m])
end

%%
model1 = 'drift';
model2 = 'stim';

cc = cc + 1;
if cc > NC
    cc = 1;
end

figure(2); clf
subplot(2,1,1)
plot(Robs(good_inds,cc), 'k'); hold on
plot(Rpred.(model1).Rpred(cc,:))
plot(Rpred.(model2).Rpred(cc,:))

subplot(2,1,2)
rfun = @(x) imboxfilt( (x(:) - Robs(good_inds,cc)).^2, 401);
plot(rfun(Rpred.(model1).Rpred(cc,:))); hold on
plot(rfun(Rpred.(model2).Rpred(cc,:)));

iix = find(dfs(good_inds,cc));

rsquared(Rpred.(model2).Rpred(cc,iix), Robs(good_inds(iix),cc))

plot(iix, Robs(good_inds(iix), cc))
plot(iix, Rpred.(model1).Rpred(cc,iix))
plot(iix, Rpred.(model2).Rpred(cc,iix))

%% empirical nonlinearity analysis
nBins = 20;

tread_speed = Stim.tread_speed(:);
isrunning = tread_speed > 1;


gains = nan(NC,1);
offsets = nan(NC,1);
empNLR = nan(NC, nBins);
empNLS = nan(NC, nBins);
bs = Stim.bin_size;

for cc = 1:NC
    
    try
        R = Robs(good_inds,cc);
        if sum(R)==0 || sum(allrate(:,cc)) == 0
            continue
        end
        % get binEdges
        [~, ~, ~, binEdges] = empiricalNonlinearity(R, allrate(:,cc), nBins);
        
        run_inds = find(tread_speed(good_inds)> 3);
        [sprateR, ~, ~] = empiricalNonlinearity(R(run_inds), allrate(run_inds,cc), binEdges);
        
        stat_inds = find(tread_speed(good_inds) < 1);
        [sprateS, ~, ~] = empiricalNonlinearity(R(stat_inds), allrate(stat_inds,cc), binEdges);
        
        X = [sprateS(:)/bs ones(nBins,1)];
        w = X\(sprateR/bs);
        
        gains(cc) = w(1);
        offsets(cc) = w(2);
        empNLR(cc,:) = sprateR;
        empNLS(cc,:) = sprateS;
    end
    
end

Running = struct();
Running.empiricalNonlinearity = struct();
Running.empiricalNonlinearity.gains = gains;
Running.empiricalNonlinearity.offsets = offsets;
Running.empiricalNonlinearity.empNLR = empNLR;
Running.empiricalNonlinearity.empNLS = empNLS;
% %%
% figure(1); clf
% subplot(1,2,1)
% 
% iix = ~isnan(gains);
% errorbar(1:nBins, nanmean(empNLR), nanstd(empNLR)/sqrt(sum(iix))); hold on
% errorbar(1:nBins, nanmean(empNLS), nanstd(empNLS)/sqrt(sum(iix))); hold on
% 
% figure(2); clf
% subplot(3,3,[2 3 5 6])
% plot(gains, offsets, '.'); hold on
% plot(xlim, [0 0], 'k')
% plot([1 1], ylim, 'k')
% xd = xlim;
% 
% subplot(3,3,[8 9])
% histogram(gains, 100); hold on
% plot([1 1], ylim)
% xlim(xd)
% 


% %%
% tread_speed = Stim.tread_speed(:);
% isrunning = tread_speed > 1;
% 
% % tread_speed = tread_speed(good_inds);
% 
% 
% figure(1); clf
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end
% R = Robs(good_inds,cc);
% 
% driftrate = Rpred.drift.Rpred(cc,:)';
% stimrate = Rpred.stim.Rpred(cc,:)';
% 
% % r0 = ~dataIdxs'*Rpred.(model).Offset(:,cc);
% % R = R - (driftrate(:,cc)+r0);
% 
% nBins = 20;
% [sprate, binCenters, spraw, binEdges, praw, pspk] = empiricalNonlinearity(R, stimrate, nBins);
% 
% run_inds = find(isrunning(good_inds));
% [sprateR, ~, sprawR] = empiricalNonlinearity(R(run_inds), stimrate(run_inds), binEdges);
% [sprateRd, ~, ~] = empiricalNonlinearity(driftrate(run_inds), stimrate(run_inds), binEdges);
% 
% stat_inds = find(~isrunning(good_inds));
% [sprateS, ~, sprawS] = empiricalNonlinearity(R(stat_inds), stimrate(stat_inds), binEdges);
% [sprateSd, ~, ~] = empiricalNonlinearity(driftrate(stat_inds), stimrate(stat_inds), binEdges);
% 
% sratesd = cellfun(@(x) std(x)/sqrt(numel(x)), spraw);
% srateRsd = cellfun(@(x) std(x)/sqrt(numel(x)), sprawR);
% srateSsd = cellfun(@(x) std(x)/sqrt(numel(x)), sprawS);
% 
% bs = Stim.bin_size;
% 
% subplot(1,2,1)
% errorbar(binCenters, sprate/bs, sratesd/bs); hold on
% errorbar(binCenters, sprateR/bs, srateRsd/bs);
% errorbar(binCenters, sprateS/bs, srateSsd/bs);
% xlabel('Generator')
% ylabel('Firing Rate')
% title(cc)
% 
% A = sprateS(:);
% B = sprateR(:);
% X = [A/bs ones(numel(A),1)];
% w = X\(B/bs);
% 
% plot(binCenters, A/bs*w(1) + w(2), 'k')
% 
% 
% subplot(1,2,2)
% errorbar(binCenters, (sprateR - sprateRd)/bs, srateRsd/bs); hold on
% errorbar(binCenters, (sprateS - sprateSd)/bs, srateSsd/bs);
% 
% 
% 
% xlabel('Generator')
% ylabel('Firing Rate')
% title(w)
% 
% % 
% % errorbar(1:numel(binCenters), sprate/bs, sratesd/bs); hold on
% % errorbar(1:numel(binCenters), sprateR/bs, srateRsd/bs);
% % errorbar(1:numel(binCenters), sprateS/bs, srateSsd/bs);
% % legend({'All', 'Running', 'Stationary'})
% % xlabel('Generator Bin ID')
% % ylabel('Firing Rate')
% % 
% % 
% % 
% %%
% model1 = 'stimsac';
% model2 = 'full';
% 
% m1 = Rpred.(model1).Rsquared;
% % m2 = Rpred.(model2).Rsquared;
% 
% figure(2); clf
% plot(imboxfilt(R,51)/bs); hold on
% plot((driftrate(:))/bs)
% plot(Rpred.full.Rpred(cc,:)/bs)
% plot(Rpred.stimsac.Rpred(cc,:)/bs)
% plot(tread_speed(good_inds))
% title(rsquared(R/bs, Rpred.RunningGain.Rpred(cc,:)/bs))
% 
% 
% title([m1(cc) m2(cc)])
% %
% tread_speed = Stim.tread_speed(:);
% tread_speed = tread_speed(good_inds);
% 
% runUp = [200 204 213];
% runDown = 202;
% 
% figure(1); clf
% plot(tread_speed); hold on
% plot(Robs(good_inds, runUp(1)))
% plot(isrunning(good_inds)); hold on
% 
% %%
% model1 = 'stimsac';
% model2 = 'full';
% 
% m1 = Rpred.(model1).Rsquared;
% m2 = Rpred.(model2).Rsquared;
% 
% [m1(cc) m2(cc)]
% %%


%% two model comparison

% model1 = 'nostim';
% model2 = 'stimrunsac';
% 
% m1 = Rpred.(model1).Rsquared;
% m2 = Rpred.(model2).Rsquared;
% 
% figure(1); clf
% cmap = lines;
% plot(m1, 'o', 'MarkerFaceColor', cmap(1,:))
% hold on
% plot(m2, 'o', 'MarkerFaceColor', cmap(1,:))
% 
% legend({model1, model2})
% 
% ylabel('Var Explained')
% xlabel('Neuron')
% 
% figure(2); clf
% plot(ecc, m2 - m1, '.'); hold on
% plot(xlim, [0 0], 'k')
% xlabel('Eccentricity')
% ylabel('\Delta R^2')
% 
% figure(3); clf
% ecc_ctrs = [1 20];
% [~, id] = min(abs(ecc(:) - ecc_ctrs), [], 2);
% cmap = lines;
% clear h
% for i = unique(id)'
%     h(i) = plot(m1(id==i), m2(id==i), 'ow', 'MarkerFaceColor', cmap(i,:)); hold on
%     plot(xlim, xlim, 'k')
%     xlabel(sprintf('Model [%s] r^2', model1))
%     ylabel(sprintf('Model [%s] r^2', model2))
% end
% legend(h, {'foveal', 'peripheral'})
% xlim(max(xlim, 0))
% ylim(max(ylim, 0))
% 
% iix = max(m1, m2) > .05;
% [pval, h, stats] = ranksum(m1(iix), m2(iix));

%% plot running
% 
% model1 = 'stimrun';
% model2 = 'stimrunsac';
% 
% BrunIdx = Rpred.(model2).covIdx == find(strcmp(Rpred.(model2).Labels, 'Running'));
% 
% Brun2 = cell2mat(cellfun(@(x) reshape(x(BrunIdx,:), [], 1)', Rpred.(model2).Beta(:), 'uni', 0));
% Brun2 = reshape(Brun2, [numel(Rpred.(model2).Beta), sum(BrunIdx), NC]);
% 
% 
% %%
% figure(1); clf
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end
% 
% 
% r2delta = Rpred.stimrunsac.Rsquared ./ Rpred.stimsac.Rsquared;
% 
% subplot(1,4,1:3)
% plot(opts.run_ctrs*Stim.bin_size, Brun2(:,1:end-1,cc)'/Stim.bin_size)
% xlabel('Time from running onset')
% title(sprintf('Unit %d', cc))
% 
% subplot(1,4,4)
% plot(Brun2(:,end,cc), 'o')
% title([Rpred.stimrunsac.Rsquared(cc) Rpred.stimsac.Rsquared(cc) r2delta(cc)])
% 
% %%
% 
% nfolds = size(Brun2,1);
% Running.run_onset_net_beta = zeros(nfolds, NC);
% Running.run_net_beta = zeros(nfolds, NC);
% Running.run_spikes_per_sec_total = zeros(nfolds, NC);
% 
% for cc = 1:NC
%     Running.run_onset_net_beta(:,cc) = mean( Brun2(:,1:end-1,cc),2) / Stim.bin_size;
%     Running.run_net_beta(:,cc) = Brun2(:,end,cc) / Stim.bin_size;
%     Running.run_spikes_per_sec_total(:,cc) = Running.run_onset_net_beta(:,cc) + Running.run_net_beta(:,cc);
% end
% 
% ecc_ctrs = [1 20];
% [~, id] = min(abs(ecc(:) - ecc_ctrs), [], 2);
% figure(2); clf
% x = mean(Running.run_spikes_per_sec_total);
% clear h
% subplot(2,1,1)
% for i = unique(id)'
%     h(i) = histogram(x(id==i), 'binEdges', -15:.5:15, 'FaceColor', cmap(i,:)); hold on
% end
% legend(h, {'foveal', 'peripheral'})
% title('All units')
% 
% subplot(2,1,2)
% clear h
% for i = unique(id)'
%     iix = r2delta > 0 & Rpred.stimrunsac.Rsquared > 0.05;
%     h(i) = histogram(x(id==i & iix(:)), 'binEdges', -15:.5:15, 'FaceColor', cmap(i,:)); hold on
% end
% legend(h, {'foveal', 'peripheral'})
% xlabel('Change in firing rate from running')
% title('cv r^2 > 0.05')
%% Get PSTH
model = 'drift';
model2 = 'stim';
Rhat = Rpred.(model).Rpred';
Rhat2 = Rpred.(model2).Rpred';

nbins = numel(Stim.trial_time);
B = eye(nbins);

dirs = max(filter([1; -1], 1, Stim.direction(:)==opts.directions(:)'), 0);
% dirs = double([zeros(1,opts.nd); diff(Stim.direction(:)==opts.directions(:)')==1]);
tax = Stim.trial_time;

n = sum(tax<=0);
Xt = temporalBases_dense(circshift(dirs, -n), B);


XY = (Xt(good_inds,:)'*Robs(good_inds,:)) ./ sum(Xt(good_inds,:))';
XYhat = (Xt(good_inds,:)'*Rhat) ./ sum(Xt(good_inds,:))';
XYhat2 = (Xt(good_inds,:)'*Rhat2) ./ sum(Xt(good_inds,:))';
%%
tax = Stim.trial_time;

Xbar = reshape(XYhat, [nbins, opts.nd, NC]);
Xbar2 = reshape(XYhat2, [nbins, opts.nd, NC]);
Rbar = reshape(XY, [nbins, opts.nd, NC]);

% 
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end

%%
figure(1); clf
subplot(1,2,1)
imagesc(tax, opts.directions, Xbar(:,:,cc)')
subplot(1,2,2)
imagesc(tax, opts.directions, Rbar(:,:,cc)')

figure(2); clf
subplot(1,2,2)
plot(tax, Rbar(:,:,cc))
yd = ylim;
xlim(tax([1 end]))


rbar = nanmean(reshape(Rbar(:,:,cc), [], 1));
r2 = rsquared(Rbar(:,:,cc), Xbar(:,:,cc), true, rbar);
r2m2 = rsquared(Rbar(:,:,cc), Xbar2(:,:,cc), true, rbar);

title(sprintf('%02.2f, %02.2f', r2, r2m2))

subplot(1,2,1)
plot(tax, Xbar(:,:,cc))
ylim(yd)
xlim(tax([1 end]))

title(sprintf('%02.2f, %02.2f', Rpred.(model).Rsquared(cc), Rpred.(model2).Rsquared(cc)))

% %% PSTH model comparison
%  
% Xbar = reshape(XYhat, [nbins, opts.nd, NC]);
% Xbar2 = reshape(XYhat2, [nbins, opts.nd, NC]);
% Rbar = reshape(XY, [nbins, opts.nd, NC]);
%     
% r2 = nan(NC, 1);
% r2m2 = nan(NC,1);
% for cc = 1:NC
%     
%     rbar = nanmean(reshape(Rbar(:,:,cc), [], 1));
%     r2(cc) = rsquared(Rbar(:,:,cc), Xbar(:,:,cc), true, rbar);
%     r2m2(cc) = rsquared(Rbar(:,:,cc), Xbar2(:,:,cc), true, rbar);
% end

% figure(1); clf
% plot(r2, r2m2, '.')
% hold on
% plot(xlim, xlim)
% 
% %%
% 
% 
% Rhat = Rpred.full.Rpred' - Rpred.stimsac.Rpred';
% XYhat = (Xt(good_inds,:)'*Rhat) ./ sum(Xt(good_inds,:))';
% Xbar = reshape(XYhat, [nbins, opts.nd, NC]);
% 
% %%
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end
% 
% 
% figure(1); clf
% plot(tax, Xbar(:,:,cc)/Stim.bin_size)

% 
% 
% %%
% r2psth = zeros(NC, 1);
% for cc = 1:NC
%     rbar = nanmean(reshape(Rbar(:,:,cc), [], 1));
%     r2psth(cc) = rsquared(Rbar(:,:,cc), Xbar(:,:,cc), true, rbar);
% end
% 
% Rpred.(model).r2psth = r2psth;

