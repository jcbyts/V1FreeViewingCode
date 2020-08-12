
load ~/Downloads/testdata.mat

addpath ~/Dropbox/MatlabCode/Repos/NIMclass/
addpath ~/Dropbox/MatlabCode/Repos/sNIMclass/
addpath(genpath('~/Dropbox/MatlabCode/Repos/L1General/'))
addpath(genpath('~/Dropbox/MatlabCode/Repos/minFunc_2012/'))
%% try NIM

dims = double([NX NY]);

nlags = 15;
numsaclags = 30;
fprintf('Building time-embedded stimulus\n')


spar = NIM.create_stim_params([nlags dims]); %, 'tent_spacing', 1);
sacpar = NIM.create_stim_params([numsaclags 1]);
X = NIM.create_time_embedding(stim, spar);
Xsac = NIM.create_time_embedding(sacoff,sacpar); 
 
par = [spar, sacpar];

%% simple LN model

LN0  = NIM( par, {'lin'}, 1,...
    'xtargets', 1,...
    'spkNL', 'softplus',...
    'd2t', 1e-2, ...
    'l1', 1e-5, ...
    'd2x', 1e-5);



%%

 

cc = 27;

r = double(Robs(:,cc));

figure(1); clf

subplot(4,1,1:3)

imagesc(r(reshape(Ti, [], num_repeats))')

subplot(4,1,4)

plot(mean(r(reshape(Ti, [], num_repeats))'))

axis tight

 

%% fit

train_inds = double(Ui(:))+1;
test_inds = double(Xi(:))+1;

LN = LN0.fit_filters(r, {X, Xsac}, train_inds);

%% 

LN2 = LN.add_subunits({'lin'}, 1, 'xtargs', 2);

LN2.subunits(2).reg_lambdas.d2t=1;
LN2.fit_filters(r, {X, Xsac}, train_inds)
%%

[LLm, rhat] = LN.eval_model(r, {X, Xsac});
[LLm2, rhat2] = LN2.eval_model(r, {X, Xsac});

 

figure(1); clf
rtrue = r(reshape(Ti+1, [], num_repeats))';
rhat = rhat(reshape(Ti+1, [],num_repeats))';
rhat2 = rhat2(reshape(Ti+1, [], num_repeats))';

plot(mean(rtrue)); hold on
plot(mean(rhat))
plot(nanmean(rhat2))

 
%%
rsquared(mean(rtrue), mean(rhat))
rsquared(mean(rtrue), mean(rhat2))

 
figure(2); clf
subplot(2,1,1)
imagesc(rtrue)
subplot(2,1,2)
imagesc(rhat)

%%
train_inds = double(Ui(:))+1;
test_inds = double(Xi(:))+1;

LN0 = prf.fit_LN(r, {stim, sacon}, par, train_inds, test_inds, 'jnk');

%%
opts = struct('num_lags_sac_pre', 15, 'fs_spikes', 120);
sacshift = [sacon(opts.num_lags_sac_pre:end); zeros(opts.num_lags_sac_pre-1,1)];

[O, OG, Xstims] = prf.fit_add_offset_gain(LN0, r, {stim, sacshift}, opts, train_inds, test_inds, 'jnk');

%%
OG.display_model

[LL0, ~] = LN0.eval_model(r, {stim, sacon}, test_inds);
[~, rhat] = LN0.eval_model(r, {stim, sacon});

[LLO] = O.eval_model(r, Xstims, test_inds);
[~, rhatO] = O.eval_model(r, Xstims);

[LLOG] = OG.eval_model(r, Xstims, test_inds);
[~, rhatOG] = OG.eval_model(r, Xstims);



figure(1); clf
rtrue = r(reshape(Ti+1, [], num_repeats))';
rhat = rhat(reshape(Ti+1, [],num_repeats))';
rhatO = rhatO(reshape(Ti+1, [], num_repeats))';
rhatOG = rhatOG(reshape(Ti+1, [], num_repeats))';

plot(mean(rtrue)); hold on
plot(mean(rhat))
plot(mean(rhatO))
plot(mean(rhatOG))

rsquared(mean(rtrue), mean(rhat))
rsquared(mean(rtrue), mean(rhatO))
rsquared(mean(rtrue), mean(rhatOG))
% rsquared(mean(rtrue), mean(rhat2))

%%
[~, rhat] = LN0.eval_model(r, {stim, sacon});
[~, rhatO] = O.eval_model(r, Xstims);
[~, rhatOG] = OG.eval_model(r, Xstims);

rbar = mean(r(train_inds));
bps0 = bitsPerSpike(rhat(test_inds), r(test_inds), rbar);
bps1 = bitsPerSpike(rhatO(test_inds), r(test_inds), rbar);
bps2 = bitsPerSpike(rhatOG(test_inds), r(test_inds), rbar);

disp([bps0, bps1, bps2])


disp([LLO-LL0 LLOG-LL0])