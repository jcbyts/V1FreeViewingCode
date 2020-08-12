% addpath ~/Dropbox/MatlabCode/Repos/ncclabcode/
% requires ncclabcode

nx=500;
ns=1e3;
X=randn(ns,nx);
xx=linspace(-1,1,nx)';
wtrue=exp(-xx.^2/.1).*cos(xx*10);
nsevar=5;
Y= X*wtrue + nsevar*randn(ns,1);


% easiest way to call function
wridge =doRegression(X,Y,'RIDGE');

% faster - preload covariance matrix
dspec.XX=X'*X;
dspec.XY=X'*Y;
dspec.YY=Y'*Y;
dspec.ntY=numel(Y);
dspec.param.regressionMode='RIDGE';

wridge2=doRegression(dspec.XX, dspec.XY, dspec.YY, dspec.ntY, dspec);


wsard=doRegression(X,Y,'SMOOTH ARD');

figure(1); clf
plot(xx, wtrue, 'k', xx, wridge.khat, xx, wsard.khat)
legend({'true', 'ridge', 'Smooth ARD'})
