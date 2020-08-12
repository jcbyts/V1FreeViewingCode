
%% paths
addpath ~/Dropbox/MatlabCode/Repos/NIMclass/
addpath ~/Dropbox/MatlabCode/Repos/sNIMclass/
addpath(genpath('~/Dropbox/MatlabCode/Repos/L1General/'))
addpath(genpath('~/Dropbox/MatlabCode/Repos/minFunc_2012/'))
        

%%
load testdata.mat

%% try NIM
dims = double([NX NY]);
nlags = 15;
fprintf('Building time-embedded stimulus\n')
tic
spar = NIM.create_stim_params([nlags dims]); %, 'tent_spacing', 1);
X = NIM.create_time_embedding(stim, spar);

%% simple LN model
LN0  = NIM( spar, {'lin'}, 1,...
    'xtargets', 1,...
    'spkNL', 'softplus',...
    'd2t', 1e-2, ...
    'l1', 1e-5, ...
    'd2x', 1e-5);
%%

cc = cc+1;
if cc > size(Robs,2)
    cc = 1;
end
r = double(Robs(:,cc));
figure(1); clf
subplot(4,1,1:3)
imagesc(r(reshape(Ti, [], num_repeats))')
subplot(4,1,4)
plot(mean(r(reshape(Ti, [], num_repeats))'))
axis tight

%% fit
train_inds = double(Ui(:))+1;
LN = LN0.fit_filters(r, {X}, train_inds);

%%
[LLm, rhat] = LN.eval_model(r, X);

figure(1); clf
rtrue = r(reshape(Ti+1, [], num_repeats))';
rhat = rhat(reshape(Ti+1, [],num_repeats))';

plot(mean(rtrue)); hold on
plot(mean(rhat))

rsquared(mean(rtrue), mean(rhat))


figure(2); clf
subplot(2,1,1)
imagesc(rtrue)
subplot(2,1,2)
imagesc(rhat)


