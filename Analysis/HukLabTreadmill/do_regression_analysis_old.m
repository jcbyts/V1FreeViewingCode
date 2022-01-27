function [Stim, opts, Rpred, Running] = do_regression_analysis(D)
% [Stim, opts, Rpred, Running] = do_regression_analysis(D)

%% Get all relevant covariates

Stim = convert_Dstruct_to_trials(D, 'bin_size', 1/60, 'pre_stim', .2, 'post_stim', .2);

%% bin spikes

bin_times = reshape(Stim.trial_time(:) + Stim.grating_onsets', [], 1);
tstart = bin_times(1);

bin_times = bin_times - tstart;
[bsorted, ind] = sort(bin_times);

if ~isfield(D, 'units')
    rfcids = [];
else
    rfcids = find(arrayfun(@(x) x.maxV > 10 & x.area > .5 , D.units));
end


spikeTimes = D.spikeTimes - tstart;
spikeIds = D.spikeIds + 1;
cids = unique(spikeIds);
if isfield(D, 'unit_area')
    cids = intersect(cids, find(strcmp(D.unit_area, 'VISp')));
end

iix = ismember(D.spikeIds, cids);
iix = spikeTimes > min(bin_times(:)) & iix;
sp = struct('st', spikeTimes(iix), 'clu', spikeIds(iix), 'cids', unique(spikeIds));

Robs = binNeuronSpikeTimesFast(sp, bsorted, Stim.bin_size*1.1);

rate_cids = find((mean(Robs) / Stim.bin_size) > 1);

cids = union(rate_cids, rfcids);
NC = numel(cids);
fprintf('%d units meet the firing rate or RF criterion \n', NC)

Robs = Robs(:, cids);
Robs = Robs(ind,:);
cc = 0;

Robs = filtfilt(ones(2,1)/2, 1, Robs);


%% get unit eccentricity
ecc = nan(1, NC);
if isfield(D, 'units')
    eccrf = arrayfun(@(x) hypot(x.center(1), x.center(2)), D.units(rfcids));
    ecc(ismember(cids, rfcids)) = eccrf;
end
%% get datafilters (when RF is on screen)

% xctr = arrayfun(@(x) x.center(1), D.units(cids));
% yctr = arrayfun(@(x) x.center(2), D.units(cids));
% xpos = reshape(Stim.eye_pos(:,:,1), [], 1) + xctr;
% ypos = reshape(Stim.eye_pos(:,:,2), [], 1) + yctr;
% 
% figure(1); clf
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end
% h = plot(xpos(:,cc), ypos(:,cc), '.');
% hold on
% h(2) = plot(D.screen_bounds([1 3]), D.screen_bounds([2 2]), 'r');
% plot(D.screen_bounds([1 3]), D.screen_bounds([4 4]), 'r')
% plot(D.screen_bounds([1 1]), D.screen_bounds([2 4]), 'r')
% plot(D.screen_bounds([3 3]), D.screen_bounds([2 4]), 'r')
% xlabel('Position (degrees)')
% ylabel('Position (degrees)')
% title('RF position')
% legend(h, {'RF location', 'Screen Boundary'})
% 
% dfs = ~(xpos < D.screen_bounds(1) | xpos > D.screen_bounds(3) | ypos < D.screen_bounds(2) | ypos > D.screen_bounds(4));
% 
% assert(all(mean(dfs) > .98), 'RFs go off the screen more than 2% of the time')



%% build the design matrix
opts = struct();

opts.stim_dur = median(D.GratingOffsets-D.GratingOnsets) + 0.05; % length of stimulus kernel
opts.spd_ctrs = [1 2];
opts.sf_ctrs = [1 3];
opts.use_onset_only = false;
opts.use_derivative = true;
opts.use_sf_tents = false;
opts.include_onset = true;
opts.include_full_stim_split = false;
opts.dph = 45; % spacing for phase basis
opts.collapse_speed = true;
opts.collapse_phase = true;
opts.rf_eccentricity = ecc;

opts.phase_ctrs = 0:opts.dph:360-opts.dph;


% running parameters
opts.run_thresh = 3;
opts.nrunbasis = 25;
opts.run_offset = -5;
opts.run_post = 40;

% saccade parameters
opts.nsacbasis = 20;
opts.sac_offset = -10;
opts.sac_post = 40;
opts.sac_covariate = 'onset';

opts.num_drift_tents = 15;

opts.nphi = numel(opts.phase_ctrs);
opts.nspd = numel(opts.spd_ctrs);

if opts.collapse_speed
    opts.nspd = 1; % collapse across speed
end
if opts.collapse_phase
    opts.nphi = 1; % collapse across phase
end

circdiff = @(x,y) angle(exp(1i*(x - y)/180*pi))/pi*180;

contrast = Stim.contrast(:);
freq = Stim.freq(:);
NT = numel(freq);

stim_on = contrast > 0 & freq > 0;

% onset only
if opts.use_onset_only
    stim_on = [false; diff(stim_on)>0];
end

direction = Stim.direction(:);
speed = Stim.speed_grating(:) .* stim_on;
speedeye = speed + Stim.speed_eye(:) .* stim_on;

opts.directions = unique(direction(stim_on));

opts.nd = numel(opts.directions);

% turn direction into "one hot" matrix
xdir = (direction == opts.directions(:)') .* stim_on;

if opts.use_sf_tents
    % spatial frequency (with basis)
    sf = tent_basis(freq, opts.sf_ctrs) .* stim_on;
    opts.nsf = numel(opts.sf_ctrs);
else
    % spatial frequency (no tents)
    opts.sf_ctrs = unique(freq(stim_on));
    opts.nsf = numel(opts.sf_ctrs);
    sf = (freq == opts.sf_ctrs(:)') .* stim_on;
end

spd = tent_basis(speedeye, opts.spd_ctrs) .* stim_on;

% figure(1); clf; plot(max(1- abs(circdiff((0:360)',phase_ctrs)/dph), 0))

xphase = max( 1 - abs(circdiff(Stim.phase_eye(:) + Stim.phase_grating(:), opts.phase_ctrs)/opts.dph), 0);

Xbig = zeros(NT, opts.nd, opts.nsf, opts.nphi, opts.nspd);
for idir = 1:opts.nd
    for isf = 1:opts.nsf
        if opts.nphi > 1
            for iphi = 1:opts.nphi
                if opts.nspd > 1
                    Xbig(:,idir, isf, iphi, :) = xdir(:,idir).*sf(:,isf).*xphase(:,iphi).*spd;
                else
                    Xbig(:,idir, isf, iphi, 1) = xdir(:,idir).*sf(:,isf).*xphase(:,iphi);
                end
            end
        else
            Xbig(:,idir, :, 1,1) = xdir(:,idir).*sf;
        end 
    end
end

Xbig = reshape(Xbig, NT, []);

if opts.use_derivative
    Xbig = max(filter([1; -1], 1, Xbig), 0);
end

figure(1); clf
imagesc(Xbig(1:300,:))
ylabel('Sample #')
xlabel('Stim Covariate')

% time embedding

stim_dur = ceil((opts.stim_dur)/Stim.bin_size);
opts.stim_ctrs = [0:10 15:5:stim_dur-2];
Bt = tent_basis(0:stim_dur, opts.stim_ctrs);

figure(1); clf, 
subplot(1,2,1)
plot((0:stim_dur)*Stim.bin_size, Bt)
title('Stimulus Temporal Basis')
xlabel('Time (bins)')

subplot(1,2,2)
imagesc(Bt)
title('Stimulus Temporal Basis')
xlabel('Basis Id')
ylabel('Time (bins)')

nlags = size(Bt,2);
Xstim = temporalBases_dense(Xbig, Bt);

% build running

Xdrift = tent_basis(1:NT, linspace(1, NT, opts.num_drift_tents));


opts.run_ctrs = linspace(opts.run_offset, opts.run_post, opts.nrunbasis);
run_basis = tent_basis(opts.run_offset:opts.run_post, opts.run_ctrs);

opts.sac_ctrs = linspace(opts.sac_offset, opts.sac_post, opts.nsacbasis);
sac_basis = tent_basis(opts.sac_offset:opts.sac_post, opts.sac_ctrs);

Xsac = zeros(NT, opts.nsacbasis);
Xrun = zeros(NT, opts.nrunbasis);

T = numel(Stim.trial_time);
num_trials = numel(Stim.grating_onsets);

for itrial = 1:num_trials
    % saccades
    if strcmp(opts.sac_covariate, 'onset')
        x = conv2(Stim.saccade_onset(:,itrial), sac_basis, 'full');
    else
        x = conv2(Stim.saccade_offset(:,itrial), sac_basis, 'full');
    end
    x = circshift(x, opts.sac_offset, 1);
    
    Xsac((itrial-1)*T + (1:T),:) = x(1:T,:);
    
    % running
    run_onset = max(filter([1; -1], 1, Stim.tread_speed(:,itrial)>opts.run_thresh), 0);
    x = conv2(run_onset, run_basis, 'full');
    x = circshift(x, opts.run_offset, 1);
    
    Xrun((itrial-1)*T + (1:T),:) = x(1:T,:);
    
end


good_inds = find(~isnan(Stim.eye_pos_proj(:)) & ~isnan(sum(Xrun,2)));

sta = (Xsac(good_inds,:)'*Robs(good_inds,:))./sum(Xsac(good_inds,:))';
rta = (Xrun(good_inds,:)'*Robs(good_inds,:))./sum(Xrun(good_inds,:))';

figure(1); clf
subplot(1,2,1)
plot(opts.sac_ctrs*Stim.bin_size, sta)
title('saccades')
subplot(1,2,2)
plot(opts.run_ctrs*Stim.bin_size, rta)
title('running')

isrunning = Stim.tread_speed(:)>opts.run_thresh;
Xrun = [Xrun isrunning];

Xonset = temporalBases_dense(stim_on, Bt);

if opts.include_full_stim_split
    XstimR = Xstim .* (isrunning & good_inds);
    XstimS = Xstim .* (~isrunning & good_inds);
    assert(sum(sum(XstimR(:,:),2)>0 & sum(XstimS(:,:),2))==0, 'Stimulus is labeled running and not running at the same time')
end

% %% double check running split
% inds = 1:300;
% 
% %%
% inds = inds + 300;
% figure(1); clf
% subplot(1,3,1)
% imagesc(XstimR(inds,:))
% subplot(1,3,2)
% imagesc(XstimS(inds,:))
% subplot(1,3,3)
% plot(sum(XstimR(inds,:),2)); hold on
% plot(sum(XstimS(inds,:),2));
% 
% plot(sum(XstimR(inds,:),2)>0 & sum(XstimS(inds,:),2), '.');


%% concatenate design matrix

if opts.include_full_stim_split
    % Add stimulus covariates
    regLabels = {'Stim R', 'Stim S', 'Stim'};
    regIdx = ones(1, size(XstimR,2));
    k = 2;
    regIdx = [regIdx repmat(k, [1, size(XstimR,2)])];
    k = k + 1;
    regIdx = [regIdx repmat(k, [1, size(Xstim,2)])];

    fullR = [XstimR, XstimS, Xstim]; %, Xonset, Xdrift, Xsac, Xrun];
else
    regLabels = {'Stim'};
    k = 1;
    regIdx = repmat(k, 1, size(Xstim,2)); %#ok<REPMAT>
    fullR = Xstim;
end

if opts.include_onset
    regLabels = [regLabels {'Stim Onset'}];
    k = k + 1;
    regIdx = [regIdx repmat(k, [1, size(Xonset,2)])];
    fullR = [fullR Xonset];
end

% add additional covariates
regLabels = [regLabels {'Drift'}];
k = k + 1;
regIdx = [regIdx repmat(k, [1, size(Xdrift,2)])];
fullR = [fullR Xdrift];

regLabels = [regLabels {'Saccade'}];
k = k + 1;
regIdx = [regIdx repmat(k, [1, size(Xsac,2)])];
fullR = [fullR Xsac];

regLabels = [regLabels {'Running'}];
k = k + 1;
regIdx = [regIdx repmat(k, [1 size(Xrun,2)])];
fullR = [fullR Xrun];

assert(size(fullR,2) == numel(regIdx), 'Number of covariates does not match Idx')
assert(numel(unique(regIdx)) == numel(regLabels), 'Number of labels does not match')

for i = 1:numel(regLabels)
    label= regLabels{i};
    cov_idx = regIdx == find(strcmp(regLabels, label));
    fprintf('[%s] has %d parameters\n', label, sum(cov_idx))
end


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
% 
% %% fit model to spike data on full dataset
% 
% [ridgeVals, dimBeta, convergenceFailures] = ridgeMML(Robs(good_inds,:), fullR(good_inds,:), true); %get ridge penalties and beta weights.
% % treat each neuron individually
% U = eye(NC);
% 
% % ecc = arrayfun(@(x) hypot(x.center(1), x.center(2)), D.units(cids));
% % U = [ecc(:) < 2, ecc(:) > 2 & ecc(:) < 15, ecc(:) > 20];
% 
% %reconstruct data and compute R^2
% Rhat = (fullR(good_inds,reg_inds) * dimBeta) + mean(Robs(good_inds,:));
% 
% %%
% corrMat = modelCorr(Robs(good_inds,:)',Rhat',U') .^2; %compute explained variance
% rbar = nanmean(Robs(good_inds,:));
% corrMat2 = rsquared(Robs(good_inds,:), Rhat, false); %compute explained variance
% 
% figure(1); clf
% subplot(1,2,1)
% cmap = lines;
% plot(corrMat, 'o', 'MarkerFaceColor', cmap(1,:)); hold on
% plot(corrMat2, '+', 'MarkerFaceColor', cmap(2,:))
% ylabel('Var Explained')
% xlabel('Neuron')
% 
% subplot(1,2,2)
% plot(corrMat, corrMat2, 'o')
% xlabel('Correlation Coefficient squared')
% ylabel('r^2')
% hold on
% plot(xlim, xlim, 'k')
% 
% %% Get PSTH
% nbins = numel(Stim.trial_time);
% B = eye(nbins);
% 
% dirs = double([zeros(1,opts.nd); diff(Stim.direction(:)==opts.directions(:)')==1]);
% tax = Stim.trial_time;
% n = sum(tax<=0);
% Xt = temporalBases_dense(circshift(dirs, -n), B);
% 
% XY = (Xt(good_inds,:)'*Robs(good_inds,:)) ./ sum(Xt(good_inds,:))';
% XYhat = (Xt(good_inds,:)'*Rhat) ./ sum(Xt(good_inds,:))';
% cc = 0;
% %%
% tax = Stim.trial_time;
% 
% Xbar = reshape(XYhat, [nbins, opts.nd, NC]);
% Rbar = reshape(XY, [nbins, opts.nd, NC]);
% 
%  
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end
% 
% figure(1); clf
% subplot(1,2,1)
% imagesc(tax, opts.directions, Xbar(:,:,cc)')
% subplot(1,2,2)
% imagesc(tax, opts.directions, Rbar(:,:,cc)')
% 
% figure(2); clf
% subplot(1,2,2)
% plot(tax, Rbar(:,:,cc))
% yd = ylim;
% xlim(tax([1 end]))
% 
% rbar = nanmean(reshape(Rbar(:,:,cc), [], 1));
% r2 = rsquared(Rbar(:,:,cc), Xbar(:,:,cc), true, rbar);
% 
% title(r2)
% 
% 
% subplot(1,2,1)
% plot(tax, Xbar(:,:,cc))
% ylim(yd)
% xlim(tax([1 end]))
% 
% figure(3); clf
% Wstim = reshape(dimBeta(regIdx==1,:), [nlags, opts.nd, opts.nsf, opts.nphi, opts.nspd, NC]);
% for i = 1:opts.nsf
%     for j = 1:opts.nphi
%         subplot(opts.nsf, opts.nphi, (i-1)*opts.nphi + j)
%         imagesc(squeeze(mean(Wstim(:,:,i,j,:,cc),5)))
%     end
% end

   

%% Build cv indices that are trial-based
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
modelNames = {'nostim', 'stim', 'stimsac', 'stimrun', 'stimrunsac'};

excludeParams = { {'Stim'}, {'Stim R', 'Stim S', 'Running', 'Saccade'}, {'Stim R', 'Stim S', 'Running'}, {'Stim R', 'Stim S', 'Saccade'}, {'Stim R', 'Stim S'}};
Rpred = struct();
for iModel = 1:numel(modelNames)
    fprintf('Fitting Model [%s]\n', modelNames{iModel})
    modelLabels = setdiff(regLabels, excludeParams{iModel}); %#ok<*NASGU>
    evalc("[Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR(good_inds,:), Robs(good_inds,:)', modelLabels, regIdx, regLabels, folds, dataIdxs);")
    Rpred.(modelNames{iModel}).Rpred = Vfull;
    Rpred.(modelNames{iModel}).Offset = cell2mat(cellfun(@(x) x(1,:), fullBeta, 'uni', 0));
    for i = 1:numel(fullBeta)
        fullBeta{i}(1,:) = [];
    end
    Rpred.(modelNames{iModel}).Beta = fullBeta; %#ok<*NODEF>
    Rpred.(modelNames{iModel}).covIdx = fullIdx;
    Rpred.(modelNames{iModel}).Ridge = fullRidge;
    Rpred.(modelNames{iModel}).Labels = fullLabels;
    
    rbar = nanmean(Robs(good_inds,:));
    Rpred.(modelNames{iModel}).Rsquared = rsquared(Robs(good_inds,:), Vfull', false, rbar); %compute explained variance
    Rpred.(modelNames{iModel}).CC = modelCorr(Robs(good_inds,:)',Vfull,U); %compute explained variance
    fprintf('Done\n')
end


%% Fit Gain models for running and pupil

%% Get stimulus prediction to create running-dependent regressor
model = 'stimsac';

NT = numel(good_inds);
assert(NT == size(dataIdxs,2), 'Number of samples does not match')

stimrateTest = zeros(NT, NC);
stimrateTrain = zeros(NT, NC);
allrate = zeros(NT,NC);
driftrate = zeros(NT, NC);
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


XstimR = stimrateTest .* isrunning(good_inds);
XstimS = stimrateTest .* ~isrunning(good_inds);


modelName = 'full';
modelLabels = regLabels(2:end);

iix = ismember(regIdx, find(ismember(regLabels, modelLabels)));

Rhat = zeros(NC, NT);
fullBeta = cell(folds,1);
fullOffsets = cell(folds, 1);
for ifold = 1:folds
    fullBeta{ifold} = zeros(sum(iix) + 2, NC);
    fullOffsets{ifold} = zeros(1, NC);
end
    
fullRidge = zeros(1,NC);
covLabels = [{'Stim R', 'Stim S'} modelLabels];
covIdx = [1 2 regIdx(regIdx~=1) + 1];
    
for ifold = 1:folds
    fprintf('Fold %d/%d\n', ifold, folds)
    
    % get stimulus prediction using the weights from this training fold (no double-dipping)
    idx = ismember(Rpred.(model).covIdx, find(ismember(Rpred.(model).Labels, {'Stim'})));
    stimpred = fullR(good_inds,idx) * Rpred.(model).Beta{ifold}(idx,:);
    XstimR = stimpred .* isrunning(good_inds);
    XstimS = stimpred .* ~isrunning(good_inds);
    
    train_inds = find(dataIdxs(ifold,:))';
    test_inds = find(~dataIdxs(ifold,:))';
    
    % loop over cells, create cell-specific design matrix and fit
    for cc = 1:NC
        %     fprintf('Unit %d/%d\n', cc, NC)
        
        Xfull = [XstimR(:,cc) XstimS(:,cc) fullR(good_inds, iix)];
        
        if ifold==1 % get Ridge params
            [fullRidge(cc), betas, convergenceFailures] = ridgeMML(Robs(good_inds(train_inds),cc), Xfull(train_inds,:), false);
        else
            Xd = [ones(NT, 1) Xfull];
            XX = Xd'*Xd;
            C = blkdiag(0, eye(size(Xfull,2)));
            XY = (Xd'*Robs(good_inds,cc));
            betas = (XX + fullRidge(cc)*C) \ XY;
        end
        
        % test prediction
        Vfull = betas(1) + Xfull(test_inds,:) * betas(2:end);
        
        Rhat(cc,test_inds) = Vfull;
        
        %     evalc("[Vfull, Beta, ~, Idx, fullRidge(cc), Labels] = crossValModel(Xfull, Robs(good_inds,cc)', covLabels, covIdx, covLabels, folds, dataIdxs);");
        
        fullBeta{ifold}(:,cc) = betas(2:end);
        fullOffsets{ifold}(cc) = betas(1);
    end
    
end


Rpred.(modelName).Rpred = Rhat;
Rpred.(modelName).Beta = fullBeta;
Rpred.(modelName).covIdx = covIdx;
Rpred.(modelName).Ridge = fullRidge;
Rpred.(modelName).Labels = covLabels;

rbar = nanmean(Robs(good_inds,:));
Rpred.(modelName).Rsquared = rsquared(Robs(good_inds,:), Rhat', false, rbar); %compute explained variance
Rpred.(modelName).CC = modelCorr(Robs(good_inds,:)',Rhat,U); %compute explained variance
fprintf('Done\n')


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
for i = 1:npairs
    subplot(2,npairs, i)
    
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
    
    plot(m1, m2, 'ow', 'MarkerSize', 6, 'MarkerFaceColor', repmat(.5, 1, 3)); hold on
    plot(xlim, xlim, 'k')
    plot([0 inc_thresh], [inc_thresh inc_thresh], 'k--')
    plot([inc_thresh inc_thresh], [0 inc_thresh], 'k--')
    xlabel(m1name)
    ylabel(m2name)
    xlim(max(xlim, 0))
    ylim(max(ylim, 0))
    
    subplot(2,npairs,i+npairs)
    iix = m2 > inc_thresh;
    histogram(m2(iix) - m1(iix), ceil(sum(iix)/2), 'FaceColor', repmat(.5, 1, 3))
    xlabel(sprintf('%s - %s', m2name, m1name))
    xlim([-m*.5 m])
end

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

Running = struct();
Running.empiricalNonlinearity = struct();
Running.empiricalNonlinearity.gains = gains;
Running.empiricalNonlinearity.offsets = offsets;
Running.empiricalNonlinearity.empNLR = empNLR;
Running.empiricalNonlinearity.empNLS = empNLS;
%%
figure(1); clf
subplot(1,2,1)

iix = ~isnan(gains);
errorbar(1:nBins, nanmean(empNLR), nanstd(empNLR)/sqrt(sum(iix))); hold on
errorbar(1:nBins, nanmean(empNLS), nanstd(empNLS)/sqrt(sum(iix))); hold on

figure(2); clf
subplot(3,3,[2 3 5 6])
plot(gains, offsets, '.'); hold on
plot(xlim, [0 0], 'k')
plot([1 1], ylim, 'k')
xd = xlim;

subplot(3,3,[8 9])
histogram(gains, 100); hold on
plot([1 1], ylim)
xlim(xd)



%%
tread_speed = Stim.tread_speed(:);
isrunning = tread_speed > 1;

% tread_speed = tread_speed(good_inds);


figure(1); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end
R = Robs(good_inds,cc);
r0 = mean(R);
% R = R - (driftrate(:,cc)+r0);

nBins = 20;
[sprate, binCenters, spraw, binEdges, praw, pspk] = empiricalNonlinearity(R, stimrateTest(:,cc), nBins);

run_inds = find(isrunning(good_inds));
[sprateR, ~, sprawR] = empiricalNonlinearity(R(run_inds), stimrateTest(run_inds,cc), binEdges);
[sprateRd, ~, ~] = empiricalNonlinearity(driftrate(run_inds,cc)+r0, stimrateTest(run_inds,cc), binEdges);

stat_inds = find(~isrunning(good_inds));
[sprateS, ~, sprawS] = empiricalNonlinearity(R(stat_inds), stimrateTest(stat_inds,cc), binEdges);
[sprateSd, ~, ~] = empiricalNonlinearity(driftrate(stat_inds,cc)+r0, stimrateTest(stat_inds,cc), binEdges);

sratesd = cellfun(@(x) std(x)/sqrt(numel(x)), spraw);
srateRsd = cellfun(@(x) std(x)/sqrt(numel(x)), sprawR);
srateSsd = cellfun(@(x) std(x)/sqrt(numel(x)), sprawS);

bs = Stim.bin_size;

subplot(1,2,1)
errorbar(binCenters, sprate/bs, sratesd/bs); hold on
errorbar(binCenters, sprateR/bs, srateRsd/bs);
errorbar(binCenters, sprateS/bs, srateSsd/bs);
xlabel('Generator')
ylabel('Firing Rate')
title(cc)

X = [sprateS(:)/bs ones(numel(sprateS),1)];
w = X\(sprateR/bs);

plot(binCenters, sprateS/bs*w(1) + w(2), 'k')

subplot(1,2,2)
errorbar(binCenters, (sprateR - sprateRd)/bs, srateRsd/bs); hold on
errorbar(binCenters, (sprateS - sprateSd)/bs, srateSsd/bs);
xlabel('Generator')
ylabel('Firing Rate')
title(cc)

% 
% errorbar(1:numel(binCenters), sprate/bs, sratesd/bs); hold on
% errorbar(1:numel(binCenters), sprateR/bs, srateRsd/bs);
% errorbar(1:numel(binCenters), sprateS/bs, srateSsd/bs);
% legend({'All', 'Running', 'Stationary'})
% xlabel('Generator Bin ID')
% ylabel('Firing Rate')
% 
% 
% 
% model1 = 'stimsac';
% model2 = 'full';
% 
% m1 = Rpred.(model1).Rsquared;
% % m2 = Rpred.(model2).Rsquared;
% 
figure(2); clf
plot(imboxfilt(R,51)/bs); hold on
plot((driftrate(:,cc) + r0)/bs)
% plot(Rpred.full.Rpred(cc,:)/bs)
% plot(Rpred.stimsac.Rpred(cc,:)/bs)
plot(tread_speed(good_inds))
title(rsquared(R/bs, Rpred.full.Rpred(cc,:)/bs))


% title([m1(cc) m2(cc)])
ecc(cc)
%%
tread_speed = Stim.tread_speed(:);
tread_speed = tread_speed(good_inds);

runUp = [200 204 213];
runDown = 202;

figure(1); clf
plot(tread_speed); hold on
plot(Robs(good_inds, runUp(1)))
plot(isrunning(good_inds)); hold on

%%
model1 = 'stimsac';
model2 = 'full';

m1 = Rpred.(model1).Rsquared;
m2 = Rpred.(model2).Rsquared;

[m1(cc) m2(cc)]
%%


%% two model comparison

model1 = 'stimsac';
model2 = 'full';

m1 = Rpred.(model1).Rsquared;
m2 = Rpred.(model2).Rsquared;

figure(1); clf
cmap = lines;
plot(m1, 'o', 'MarkerFaceColor', cmap(1,:))
hold on
plot(m2, 'o', 'MarkerFaceColor', cmap(1,:))

legend({model1, model2})

ylabel('Var Explained')
xlabel('Neuron')

figure(2); clf
plot(ecc, m2 - m1, '.'); hold on
plot(xlim, [0 0], 'k')
xlabel('Eccentricity')
ylabel('\Delta R^2')

figure(3); clf
ecc_ctrs = [1 20];
[~, id] = min(abs(ecc(:) - ecc_ctrs), [], 2);
cmap = lines;
clear h
for i = unique(id)'
    h(i) = plot(m1(id==i), m2(id==i), 'ow', 'MarkerFaceColor', cmap(i,:)); hold on
    plot(xlim, xlim, 'k')
    xlabel(sprintf('Model [%s] r^2', model1))
    ylabel(sprintf('Model [%s] r^2', model2))
end
legend(h, {'foveal', 'peripheral'})
xlim(max(xlim, 0))
ylim(max(ylim, 0))

iix = max(m1, m2) > .05;
[pval, h, stats] = ranksum(m1(iix), m2(iix));

%% plot running

model1 = 'stimrun';
model2 = 'stimrunsac';

BrunIdx = Rpred.(model2).covIdx == find(strcmp(Rpred.(model2).Labels, 'Running'));

Brun2 = cell2mat(cellfun(@(x) reshape(x(BrunIdx,:), [], 1)', Rpred.(model2).Beta(:), 'uni', 0));
Brun2 = reshape(Brun2, [numel(Rpred.(model2).Beta), sum(BrunIdx), NC]);


%%
figure(1); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end


r2delta = Rpred.stimrunsac.Rsquared ./ Rpred.stimsac.Rsquared;

subplot(1,4,1:3)
plot(opts.run_ctrs*Stim.bin_size, Brun2(:,1:end-1,cc)'/Stim.bin_size)
xlabel('Time from running onset')
title(sprintf('Unit %d', cc))

subplot(1,4,4)
plot(Brun2(:,end,cc), 'o')
title([Rpred.stimrunsac.Rsquared(cc) Rpred.stimsac.Rsquared(cc) r2delta(cc)])

%%

nfolds = size(Brun2,1);
Running.run_onset_net_beta = zeros(nfolds, NC);
Running.run_net_beta = zeros(nfolds, NC);
Running.run_spikes_per_sec_total = zeros(nfolds, NC);

for cc = 1:NC
    Running.run_onset_net_beta(:,cc) = mean( Brun2(:,1:end-1,cc),2) / Stim.bin_size;
    Running.run_net_beta(:,cc) = Brun2(:,end,cc) / Stim.bin_size;
    Running.run_spikes_per_sec_total(:,cc) = Running.run_onset_net_beta(:,cc) + Running.run_net_beta(:,cc);
end

ecc_ctrs = [1 20];
[~, id] = min(abs(ecc(:) - ecc_ctrs), [], 2);
figure(2); clf
x = mean(Running.run_spikes_per_sec_total);
clear h
subplot(2,1,1)
for i = unique(id)'
    h(i) = histogram(x(id==i), 'binEdges', -15:.5:15, 'FaceColor', cmap(i,:)); hold on
end
legend(h, {'foveal', 'peripheral'})
title('All units')

subplot(2,1,2)
clear h
for i = unique(id)'
    iix = r2delta > 0 & Rpred.stimrunsac.Rsquared > 0.05;
    h(i) = histogram(x(id==i & iix(:)), 'binEdges', -15:.5:15, 'FaceColor', cmap(i,:)); hold on
end
legend(h, {'foveal', 'peripheral'})
xlabel('Change in firing rate from running')
title('cv r^2 > 0.05')
%% Get PSTH
model = 'stimsac';
model2 = 'full';
Rhat = Rpred.(model).Rpred';
Rhat2 = Rpred.(model2).Rpred';

nbins = numel(Stim.trial_time);
B = eye(nbins);

dirs = double([zeros(1,opts.nd); diff(Stim.direction(:)==opts.directions(:)')==1]);
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


cc = cc + 1;
if cc > NC
    cc = 1;
end


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

%% PSTH model comparison
 
Xbar = reshape(XYhat, [nbins, opts.nd, NC]);
Xbar2 = reshape(XYhat2, [nbins, opts.nd, NC]);
Rbar = reshape(XY, [nbins, opts.nd, NC]);
    
r2 = nan(NC, 1);
r2m2 = nan(NC,1);
for cc = 1:NC
    
    rbar = nanmean(reshape(Rbar(:,:,cc), [], 1));
    r2(cc) = rsquared(Rbar(:,:,cc), Xbar(:,:,cc), true, rbar);
    r2m2(cc) = rsquared(Rbar(:,:,cc), Xbar2(:,:,cc), true, rbar);
end

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

