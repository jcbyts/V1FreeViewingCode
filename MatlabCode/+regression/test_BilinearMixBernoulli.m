%% test bilinear logistic regression


ntb  = 10;
nsb  = 10;
nlin = 0;

wspace = exp(- ((1:nsb)-nsb/2).^2/15);
wtime  = cos( .5*(1:ntb));
wlin = exp(-cos(1:nlin));
wst = wtime(:)*wspace(:)';

figure(1); clf
subplot(1,3,1)
plot(wlin)
title('Linear Weights')
subplot(1,3,2)
imagesc(wst)
title('Bilinear Weights')


wtrue = [wlin(:); wst(:)]/10;
nk       = numel(wtrue); 
nsamples = 1500;
X = randn(nsamples, nk);

Y = rand(nsamples,1) < (1./(1+exp(X*-wtrue)));
%% linear bilinear regression

xx = X'*X;
xy = X'*Y;
wDims = [ntb nsb];
p = 1; % rank 1
indsbilin = (nlin+1):nk;
% [w_hat,wt,wx,wlin] = bilinearMixRegress_coordAscent(xx,xy,wDims,p,indsbilin);
lfun = @(w) neglogli_bernoulliGLM(w, X, Y);
w0 = xx\xy;
optimOpts = optimoptions('fminunc', 'algorithm', 'trust-region', 'PrecondBandWidth', 1, 'GradObj', 'on', 'Hessian', 'on');
what = fminunc(lfun, w0, optimOpts);

subplot(1,3,1); imagesc(reshape(what(nlin+1:end), [ntb nsb]))
title('MLE estimate')
%%
[w_hat,~,wt,wx,wlin] = bilinearMixRegress_Bernoulli(X,Y,wDims,p,indsbilin, [], true);

subplot(1,3,3)
imagesc(reshape(w_hat(nlin+1:end), [ntb nsb]))

%% 2D logistic AR1
% what=autoRegress_logisticRidge(X,Y,1:nk, 5, [1 10], w_hat)
% profile on
% [what, hprs, SDebars, postHess, logevid] = autoRegress_logisticAR1_2D(X,Y,[ntb nsb], .1, [.9 1 10], [.9 .99 1])
% profile viewer
%%
[wtt, ~, wxx]=svd(reshape(what, ntb, nsb));
figure(1); clf
subplot(1,3,1)
plot(wt); hold on
plot(wtime, 'k')
plot(-wtt(:,1))
subplot(1,3,2)
plot(wx); hold on
plot(wspace, 'k')
plot(wxx(:,1))
subplot(1,3,3)
plot(w_hat); hold on
plot(wtrue, 'k'); 
plot(what)

sum(sum(w_hat - wtrue).^2)
sum(sum(what - wtrue).^2)
