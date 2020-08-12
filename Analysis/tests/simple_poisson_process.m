
%% poisson process simple

n = 10e3;
lambda = .5;
nlags = 100;

y = poissrnd(lambda*ones(n,1));


figure(1); clf
[ac, lags] = xcorr(y, y, nlags);

plot(lags, ac); hold on
plot(xlim, [1 1]*(lambda^2)*(n-nlags))

figure(2); clf
K = 100;
h = histogram(diff(find(y)), 'binEdges', 0:K); hold on
plot(0:K, (lambda*n)*lambda*exp(-lambda*(0:K)))

