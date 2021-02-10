function [stat, opts] = spat_rf_helper_regress(Exp, varargin)
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
ip.addParameter('ROI', [0 -2 2 0]) %[-14 -10 14 10])
ip.addParameter('binSize', .1)
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
Xstim = makeStimRows(Stim, num_lags);
Xstim = circshift(Xstim, win(1));

% find valid fixations
valid = find(opts.eyeLabel==1);
nValid = numel(valid);
%% estimate receptive field
% assume smoothness in space and time.
% estimate the amount of smoothness by maximizing the model evidence
% using a gridsearch

% smooth spike trains
Rvalid = imgaussfilt(RobsSpace-mean(RobsSpace), [.5 0.001])*fs_stim;
% compute STA (in units of delta spike rate)
% Rvalid = RobsSpace - mean(RobsSpace);

% random sample train and test set
test = randsample(valid, floor(nValid/5));
train = setdiff(valid, test);

XX = (Xstim(train,:)'*Xstim(train,:)); % covariance
XY = (Xstim(train,:)'*Rvalid(train,:));

Rtest = Rvalid(test,:);

CpriorInv = qfsmooth3D([num_lags, fliplr(opts.dims)], [.25 .5]);
CpriorInv = CpriorInv + eye(size(CpriorInv,2));

%% loop over hyperparameter
lambdas = (2.^(-1:20));
nLambdas = numel(lambdas);

r2test = zeros(nLambdas,NC);
kbest = XY;

for il = 1:nLambdas 
    fprintf('%d/%d regularization levels\n', il, nLambdas)
    % get posterior weights
    lambda = lambdas(il);
    
    H = XX + lambda*CpriorInv;
    
    kmap = H\XY;  % Posterior Mean (MAP estimate)
    
    % predict on test set
    Rhat = Xstim(test,:)*kmap;
    
    for cc = 1:NC
        rtmp = rsquared(Rtest(:,cc), Rhat(:,cc));
        if rtmp > r2test(:,cc) % store best RF
            kbest(:,cc) = kmap(:,cc);
        end
        r2test(il,cc) = rtmp;
    end
    
end

[r2best, id] = max(r2test);
bestlambda = lambdas(id);

%%
if ip.Results.plot
    figure
end

[xx,yy]=meshgrid(opts.xax, opts.yax);

xax = 1e3*(win(1):win(2))/fs_stim;

stat.rf = reshape(kbest, [num_lags prod(opts.dims) NC]);
stat.timeax = xax;
stat.fs_stim = fs_stim;
stat.dim = opts.dims;
stat.ppd = Exp.S.pixPerDeg;
stat.roi = ip.Results.ROI;
stat.spatrf = zeros([opts.dims NC]);
stat.peaklag = zeros(NC, 1);
stat.sdbase = zeros(NC,1);
stat.lambda = bestlambda(:);
stat.r2test = r2best(:);

rfLocations = nan(NC,2);

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

for cc = 1:NC
    if ip.Results.plot
        subplot(sx, sy, cc)
    end
    
    I = stat.rf(:,:,cc);
%     I = reshape(kmap(:,cc), num_lags, []);
    
    s = std(reshape(I(xax<0,:), [], 1));
    
%     plot(I./s)
%     if r2best(cc) < 0
%         axis off
%     end
%     
%     continue
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
