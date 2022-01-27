

%% Get all relevant covariates

Stim = convert_Dstruct_to_trials(D, 'bin_size', 1/60, 'pre_stim', .2, 'post_stim', .2);

%% bin spikes

bin_times = reshape(Stim.trial_time(:) + Stim.grating_onsets', [], 1);
[bsorted, ind] = sort(bin_times);

cids = find(arrayfun(@(x) x.area > .5, D.units));

iix = ismember(D.spikeIds, cids);
sp = struct('st', D.spikeTimes(iix), 'clu', D.spikeIds(iix), 'cids', cids);

Robs = binNeuronSpikeTimesFast(sp, bsorted, Stim.bin_size*1.1);

rate_cids = find((mean(Robs) / Stim.bin_size) > 1);

cids = intersect(rate_cids, cids);
NC = numel(cids);
fprintf('%d units meet the firing rate / RF criterion \n', NC)

Robs = Robs(:, cids);
Robs = Robs(ind,:);
cc = 0;

% get unit eccentricity
ecc = arrayfun(@(x) hypot(x.center(1), x.center(2)), D.units(cids));
%% get datafilters (when RF is on screen)

xctr = arrayfun(@(x) x.center(1), D.units(cids));
yctr = arrayfun(@(x) x.center(2), D.units(cids));
xpos = reshape(Stim.eye_pos(:,:,1), [], 1) + xctr;
ypos = reshape(Stim.eye_pos(:,:,2), [], 1) + yctr;

figure(1); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end
h = plot(xpos(:,cc), ypos(:,cc), '.');
hold on
h(2) = plot(D.screen_bounds([1 3]), D.screen_bounds([2 2]), 'r');
plot(D.screen_bounds([1 3]), D.screen_bounds([4 4]), 'r')
plot(D.screen_bounds([1 1]), D.screen_bounds([2 4]), 'r')
plot(D.screen_bounds([3 3]), D.screen_bounds([2 4]), 'r')
xlabel('Position (degrees)')
ylabel('Position (degrees)')
title('RF position')
legend(h, {'RF location', 'Screen Boundary'})

dfs = ~(xpos < D.screen_bounds(1) | xpos > D.screen_bounds(3) | ypos < D.screen_bounds(2) | ypos > D.screen_bounds(4));

assert(all(mean(dfs) > .98), 'RFs go off the screen more than 2% of the time')



%% build the design matrix
opts = struct();
opts.stim_dur = 1.050; % length of stimulus kernel
opts.spd_ctrs = [1 2];
opts.sf_ctrs = [1 3];
opts.use_onset_only = false;
opts.use_derivative = true;
opts.use_sf_tents = false;
opts.include_onset = true;

opts.dph = 45; % spacing for phase basis
opts.collapse_speed = true;
opts.collapse_phase = true;

opts.phase_ctrs = 0:opts.dph:360-opts.dph;
opts.nphi = numel(opts.phase_ctrs);
opts.nspd = numel(opts.spd_ctrs);

% running parameters
opts.run_thresh = 3;
opts.nrunbasis = 20;
opts.run_offset = -5;
opts.run_post = 30;

% saccade parameters
opts.nsacbasis = 20;
opts.sac_offset = -10;
opts.sac_post = 40;
opts.sac_covariate = 'onset';


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

%% time embedding

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

%% build running

Xdrift = tent_basis(1:NT, linspace(1, NT, 10));


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


good_inds = ~isnan(Stim.eye_pos_proj(:)) & ~isnan(sum(Xrun,2));

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

XstimR = Xstim .* (isrunning & good_inds);
XstimS = Xstim .* (~isrunning & good_inds);

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


assert(sum(sum(XstimR(:,:),2)>0 & sum(XstimS(:,:),2))==0, 'Stimulus is labeled running and not running at the same time')

%% concatenate design matrix

% Add stimulus covariates
regLabels = {'Stim R', 'Stim S', 'Stim'};
regIdx = ones(1, size(XstimR,2));
k = 2;
regIdx = [regIdx repmat(k, [1, size(XstimR,2)])];
k = k + 1;
regIdx = [regIdx repmat(k, [1, size(Xstim,2)])];

fullR = [XstimR, XstimS, Xstim]; %, Xonset, Xdrift, Xsac, Xrun];

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


%% run QR and check for rank-defficiency. This will show whether a given regressor is highly collinear with other regressors in the design matrix.
% The resulting plot ranges from 0 to 1 for each regressor, with 1 being
% fully orthogonal to all preceeding regressors in the matrix and 0 being
% fully redundant. Having fully redundant regressors in the matrix will
% break the model, so in this example those regressors are removed. In
% practice, you should understand where the redundancy is coming from and
% change your model design to avoid it in the first place!
reg_inds = find(regIdx > 2);
rejIdx = false(1,size(fullR(good_inds,reg_inds),2));
[~, fullQRR] = qr(bsxfun(@rdivide,fullR(good_inds,reg_inds),sqrt(sum(fullR(good_inds,reg_inds).^2))),0); %orthogonalize normalized design matrix
figure; plot(abs(diag(fullQRR)),'linewidth',2); ylim([0 1.1]); title('Regressor orthogonality'); drawnow; %this shows how orthogonal individual regressors are to the rest of the matrix
axis square; ylabel('Norm. vector angle'); xlabel('Regressors');
if sum(abs(diag(fullQRR)) > max(size(fullR(good_inds,reg_inds))) * eps(fullQRR(1))) < size(fullR(good_inds,reg_inds),2) %check if design matrix is full rank
    temp = ~(abs(diag(fullQRR)) > max(size(fullR(good_inds,reg_inds))) * eps(fullQRR(1)));
    fprintf('Design matrix is rank-defficient. Removing %d/%d additional regressors.\n', sum(temp), sum(~rejIdx));
    rejIdx(~rejIdx) = temp; %reject regressors that cause rank-defficint matrix
end
% save([fPath filesep 'regData.mat'], 'fullR', 'regIdx', 'regLabels','fullQRR','-v7.3'); %save some model variables

%% fit model to spike data on full dataset

[ridgeVals, dimBeta] = ridgeMML(Robs(good_inds,:), fullR(good_inds,reg_inds), true); %get ridge penalties and beta weights.
% treat each neuron individually
U = eye(NC);

% ecc = arrayfun(@(x) hypot(x.center(1), x.center(2)), D.units(cids));
% U = [ecc(:) < 2, ecc(:) > 2 & ecc(:) < 15, ecc(:) > 20];

%reconstruct data and compute R^2
Rhat = (fullR(good_inds,reg_inds) * dimBeta) + mean(Robs(good_inds,:));

%%
corrMat = modelCorr(Robs(good_inds,:)',Rhat',U') .^2; %compute explained variance
rbar = nanmean(Robs(good_inds,:));
corrMat2 = rsquared(Robs(good_inds,:), Rhat, false, rbar); %compute explained variance

figure(1); clf
subplot(1,2,1)
cmap = lines;
plot(corrMat, 'o', 'MarkerFaceColor', cmap(1,:)); hold on
plot(corrMat2, '+', 'MarkerFaceColor', cmap(2,:))
ylabel('Var Explained')
xlabel('Neuron')

subplot(1,2,2)
plot(corrMat, corrMat2, 'o')
hold on
plot(xlim, xlim, 'k')

%% Get PSTH
nbins = numel(Stim.trial_time);
B = eye(nbins);

dirs = double([zeros(1,opts.nd); diff(Stim.direction(:)==opts.directions(:)')==1]);
tax = Stim.trial_time;
n = sum(tax<=0);
Xt = temporalBases_dense(circshift(dirs, -n), B);


XY = (Xt(good_inds,:)'*Robs(good_inds,:)) ./ sum(Xt(good_inds,:))';
XYhat = (Xt(good_inds,:)'*Rhat) ./ sum(Xt(good_inds,:))';
cc = 0;
%%
tax = Stim.trial_time;

Xbar = reshape(XYhat, [nbins, opts.nd, NC]);
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

title(r2)


subplot(1,2,1)
plot(tax, Xbar(:,:,cc))
ylim(yd)
xlim(tax([1 end]))

figure(3); clf
Wstim = reshape(dimBeta(regIdx==1,:), [nlags, opts.nd, opts.nsf, opts.nphi, opts.nspd, NC]);
for i = 1:opts.nsf
    for j = 1:opts.nphi
        subplot(opts.nsf, opts.nphi, (i-1)*opts.nphi + j)
        imagesc(squeeze(mean(Wstim(:,:,i,j,:,cc),5)))
    end
end

   

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
modelNames = {'full', 'stim', 'stimsac', 'stimrun', 'stimrunsac'};
excludeParams = { {'Stim'}, {'Stim R', 'Stim S', 'Running', 'Saccade'}, {'Stim R', 'Stim S', 'Running'}, {'Stim R', 'Stim S', 'Saccade'}, {'Stim R', 'Stim S'}};
Rpred = struct();
for iModel = 1:numel(modelNames)
    fprintf('Fitting Model [%s]\n', modelNames{iModel})
    modelLabels = setdiff(regLabels, excludeParams{iModel});
    [Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR(good_inds,:), Robs(good_inds,:)', modelLabels, regIdx, regLabels, folds, dataIdxs);
    Rpred.(modelNames{iModel}).Rpred = Vfull;
    Rpred.(modelNames{iModel}).Beta = fullBeta;
    Rpred.(modelNames{iModel}).covIdx = fullIdx;
    Rpred.(modelNames{iModel}).Ridge = fullRidge;
    Rpred.(modelNames{iModel}).Labels = fullLabels;
    Rpred.(modelNames{iModel}).Rsquared = modelCorr(Robs(good_inds,:)',Vfull,U) .^2; %compute explained variance
    fprintf('Done\n')
end


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
    
    subplot(2,npairs,i+npairs)
    iix = m2 > inc_thresh;
    histogram(m2(iix) - m1(iix), ceil(sum(iix)/2), 'FaceColor', repmat(.5, 1, 3))
    xlabel(sprintf('%s - %s', m2name, m1name))
    xlim([-m*.5 m])
end

%% two model comparison

model1 = 'stimsac';
model2 = 'stimrunsac';

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
plot(opts.run_ctrs*Stim.bin_size, Brun2(:,1:end-1,cc)')
xlabel('Time from running onset')
title(sprintf('Unit %d', cc))

subplot(1,4,4)
plot(Brun2(:,end,cc), 'o')
title([Rpred.stimrunsac.Rsquared(cc) Rpred.stimsac.Rsquared(cc) r2delta(cc)])


Running = struct();
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
    h(i) = histogram(x(id==i & iix), 'binEdges', -15:.5:15, 'FaceColor', cmap(i,:)); hold on
end
legend(h, {'foveal', 'peripheral'})
xlabel('Change in firing rate from running')
title('cv r^2 > 0.05')
%% Get PSTH
model = 'stimrunsac';
Rhat = Rpred.(model).Rpred';

nbins = numel(Stim.trial_time);
B = eye(nbins);

dirs = double([zeros(1,opts.nd); diff(Stim.direction(:)==opts.directions(:)')==1]);
tax = Stim.trial_time;
n = sum(tax<=0);
Xt = temporalBases_dense(circshift(dirs, -n), B);


XY = (Xt(good_inds,:)'*Robs(good_inds,:)) ./ sum(Xt(good_inds,:))';
XYhat = (Xt(good_inds,:)'*Rhat) ./ sum(Xt(good_inds,:))';
%%
tax = Stim.trial_time;

Xbar = reshape(XYhat, [nbins, opts.nd, NC]);
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

title(r2)

subplot(1,2,1)
plot(tax, Xbar(:,:,cc))
ylim(yd)
xlim(tax([1 end]))


%%
r2psth = zeros(NC, 1);
for cc = 1:NC
    rbar = nanmean(reshape(Rbar(:,:,cc), [], 1));
    r2psth(cc) = rsquared(Rbar(:,:,cc), Xbar(:,:,cc), true, rbar);
end

Rpred.(model).r2psth = r2psth;

