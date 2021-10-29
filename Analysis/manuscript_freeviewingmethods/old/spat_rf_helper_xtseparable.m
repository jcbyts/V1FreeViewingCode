function [stat, opts] = spat_rf_helper_xtseparable(Exp, varargin)
% [stat, opts] = spat_rf_helper(Exp, varargin)
% 'ROI',       [-14 -10 14 10]
% 'binSize',   1
% 'win',       [-5 15]
% 'numspace',  20
% 'plot',      true
% 'sdthresh',  4
%% New way: forward correlation

% build default options
ip = inputParser();
ip.addParameter('ROI', [-14 -10 14 10])
ip.addParameter('binSize', 1)
ip.addParameter('win', [-5 15])
ip.addParameter('numspace', 20)
ip.addParameter('plot', true)
ip.addParameter('sdthresh', 4)
ip.parse(varargin{:})


%% build stimulus matrix for spatial mapping
win = ip.Results.win;
num_lags = diff(win)+1;


eyePos = Exp.vpx.smo(:,2:3);

[Stim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', ip.Results.ROI*Exp.S.pixPerDeg, 'binSize', ip.Results.binSize*Exp.S.pixPerDeg, 'eyePos', eyePos);
fs_stim = round(1/median(diff(opts.frameTimes)));

NC = size(RobsSpace,2);

% embed time
Xstim = fliplr(makeStimRows(Stim, num_lags));
Xstim = circshift(Xstim, win(1));

% find valid fixations
valid = find(opts.eyeLabel==1);
nValid = numel(valid);
%% estimate receptive field
% assume smoothness in space and time.
% estimate the amount of smoothness by maximizing the model evidence
% using a gridsearch

% smooth spike trains
Rvalid = imgaussfilt(RobsSpace-mean(RobsSpace), [.5 0.001]);
% compute STA (in units of delta spike rate)
% Rvalid = RobsSpace - mean(RobsSpace);

% random sample train and test set
test = randsample(valid, floor(nValid/5));
train = setdiff(valid, test);

% XX = (Xstim(train,:)'*Xstim(train,:)); % covariance
XY = (Xstim(train,:)'*Rvalid(train,:));

Rtest = Rvalid(test,:);

%% get space-time separable RFs

fitopts = struct('MaxIter',50, 'TolFun',1e-6, 'Display','iter');


X = Xstim(train,:);

figure(1); clf
% wst = reshape(mean(XY,2), num_lags,[]);
wst = simpleRevcorrValid(Stim,mean(Rvalid,2),num_lags,train,win(1));

[u,~,~] = svd(wst);
wt0 = u(:,1)*sign(sum(u(:,1)));
wt0 = wt0 / norm(wt0);

Sfilt = filter(wt0, 1, Stim);
Xs = Sfilt(train,:);

wx0 = Xs'*mean(Rvalid(train,:),2);

%%
Ctinv = 5*qfsmooth1D(num_lags);
Cxinv = 50*qfsmooth2nd(opts.dims(1), opts.dims(2));
Cxinv = 100*eye(prod(opts.dims)) + Cxinv;

cc = cc + 1;
if cc > NC
    cc = 1;
end

Y = Rvalid(:,cc);
wt = repmat(wt0, 1, 1);
wx = repmat(wx0, 1, 1);
   
subplot(1,2,1)
plot(wt)
subplot(1,2,2)
imagesc(reshape(wx, opts.dims))
drawnow

% Compute initial error
Ypred = sameconv(Stim*wx,wt);  % predicted Y
Ypred = Ypred(train,:); % valid index

fval = .5*norm(Y(train)-Ypred)^2 + .5*(sum(sum((wt'*Ctinv*wt).*(wx'*Cxinv*wx))));  % value of neg log-posterior
fchange = inf;

iter = 1;
if strcmp(fitopts.Display, 'iter')
    fprintf('--- bilinearRegress_coordAscent_xtMAP: beginning coordinate descent ---\n');
    fprintf('Iter 0: fval = %.4f\n',fval);
end

while (iter <= fitopts.MaxIter) && (fchange > fitopts.TolFun)
       
    [XYt,XXt] = simpleRevcorrValid(Stim*wx,Y,num_lags,train,win(1));  % project stimulus spatially
        
    wt = (XXt+Ctinv)\XYt(:);
    wt = wt/norm(wt);
    
    % Update spatial components
    Xx = sameconvcols(Stim,wt);
    Xx = Xx(train,:);
    wx = (Xx'*Xx+Cxinv)\(Xx'*Y(train,:));
    plot(wx)

    % Compute size of change 
    Ypred = Xx*wx(:);  % predicted Y
    fvalnew = .5*norm(Y(train)-Ypred)^2+ .5*sum(sum((wt'*Ctinv*wt).*(wx'*Cxinv*wx)));  % value of MSE
    fchange = fval-fvalnew;
    fval = fvalnew;
    iter = iter+1;
    if strcmp(fitopts.Display, 'iter')
        fprintf('Iter %d: fval = %.4f,  fchange = %.4f\n',iter-1,fval,fchange);
    end
end

if iter==fitopts.MaxIter
    fprintf('bilinearRegress_coordAscent_xt: terminated because maxiter (%d) reached\n',opts.MaxIter);
end


w_hat = wt*wx';

figure(1); clf
subplot(1,2,1)
plot(wt)
subplot(1,2,2)
imagesc(reshape(wx, opts.dims))
% Xr = reshape(X, [NT, num_lags, nd])

% plot(u(:,1))
% imagesc(reshape(v(:,1), opts.dims))


%%
iter = 25;

X

%% do regression
nt = 15;
wrank = 1;
Ctinv = .1*qfsmooth1D(nt);
Cxinv = 5*qfsmooth2nd(opts.dims(1), opts.dims(2));
Cxinv = 1*eye(prod(opts.dims)) + Cxinv;
fitopts = struct('MaxIter',50, 'TolFun',1e-6, 'Display','iter');

addpath MatlabCode/deprecated/
% Rvalid = imgaussfilt(RobsSpace-mean(RobsSpace), [2 0.001]);
Rvalid = RobsSpace - mean(RobsSpace);
cc = cc + 1;
if cc > NC
    cc = 1;
end

[w_hat,wt,wx] = bilinearRegress_coordAscent_xtMAP(Stim,sum(Rvalid,2),nt,wrank,Ctinv,Cxinv,train, fitopts);

figure(1);
clf
subplot(1,wrank+1,1)
plot(wt)

for i = 1:wrank
    subplot(1,1+wrank,1+i)
    imagesc(reshape(wx(:,i), opts.dims))
end

%%
% Xfilt = S
%% do forward correlation
d = size(Xstim, 2);
NC = size(RobsSpace,2);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

win = ip.Results.win;
num_lags = diff(win)+1;
mrf = zeros(num_lags, d, NC);

% smooth spike trains
Rdelta = imgaussfilt(RobsSpace-mean(RobsSpace), [1 0.01]);

%%

% [w_hat,wt,wx] = bilinearRegress_coordAscent_xtMAP(X,Y,nt,wrank,Ctinv,Cxinv,opts)
% 
% Computes MAP linear regression estimate with a bilinear parametrization of
% a low-rank linear regression weights, using lightweight memory-efficient implementation
% that avoids computing entire x'*x matrix.
%
% Finds solution to argmin_w ||y - x*w||^2 + .5*Tr[Ctinv*w*Cxinv*w']
% where w are parametrized as vec( wt*wx')
%
% Inputs:
%   X [nt x nx] - stimulus matrix (each row is stimulus for a single time bin)
%   Y [nt x 1]  - vector of responses at each time bin
%            nt - # time bins in stim filter
%         wrank - rank of bilinear filter
%         Ctinv - inverse covariance for t params
%         Cxinv - inverse covariance for x params
%          opts - options struct (optional)
%                 fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
%%

% loop over stimulus dimensions
disp('Running forward correlation...')
for k = 1:d
%     fprintf('%d/%d\n', k, d)

%     ind = find(Xstim(:,k) > 1);
    s = fliplr(makeStimRows(Xstim(:,k), num_lags));
    s = circshift(s, win(1));
    s = s';
    an = s*Rdelta;
    
    mrf(:,k,:) = an;
end
disp('Done')

stat.rf = mrf;

%% plot / compute cell-by cell quantities

if ip.Results.plot
    figure(2); clf
end

[xx,yy]=meshgrid(opts.xax, opts.yax);

rfLocations = nan(NC,2);

fs_stim = round(1/median(diff(opts.frameTimes)));
xax = 1e3*(win(1):win(2))/fs_stim;

stat.timeax = xax;
stat.fs_stim = fs_stim;
stat.dim = opts.dims;
stat.ppd = Exp.S.pixPerDeg;
stat.roi = ip.Results.ROI;
stat.spatrf = zeros([opts.dims NC]);
stat.peaklag = zeros(NC, 1);
stat.sdbase = zeros(NC,1);

for cc = 1:NC
    
    if ip.Results.plot
        subplot(sx,sy,cc)
    end
    
    I = mrf(:,:,cc); % individual neuron STA
    
%     plot(I)
%     continue
    
    s = std(reshape(I(xax<0, :), [], 1)); %standard deviation of STA before stim onset (noise estimate)
    stat.sdbase(cc) = s;
    
    t = abs(I./s) > ip.Results.sdthresh; % does the unit have significant excursions?
    
    if sum(t(:)) < 2
        axis off
        continue
    end
    
    % get center xy
    [i,~] = find(abs(I)==max(abs(I(:))));
    if numel(i)~=1
        disp(i)
        continue
    end
    
    
    Ispace = reshape(I(i,:), opts.dims);
    
    
    stat.peaklag(cc) = i;
    stat.spatrf(:,:,cc) = Ispace;
    
    % threshold out for 
    I(~t) = 0;
    Ispace = reshape(I(i,:), opts.dims);
    
    
    if ip.Results.plot
        imagesc(opts.xax, opts.yax, stat.spatrf(:,:,cc)); hold on
        colormap(viridis)
    end
    
    pow = 10;
    Ispace = Ispace.^pow ./ sum(Ispace(:).^pow);
    
    % get softmax center
    x0 = xx(:)'*Ispace(:);
    y0 = yy(:)'*Ispace(:);
    
    if ip.Results.plot
        plot(x0, y0, '.r')
        title(cc)
    end
    
    rfLocations(cc,:) = [x0 y0];

end

stat.rfLocations = rfLocations;