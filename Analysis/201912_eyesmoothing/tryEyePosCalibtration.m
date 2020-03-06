
%% add paths

user = 'jakework';
addFreeViewingPaths(user);

switch user
    case 'jakework'
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\NIMclass
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\sNIMclass
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\L1General'))    
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\minFunc_2012'))  
    case 'jakelaptop'
        addpath ~/Dropbox/MatlabCode/Repos/NIMclass/
        addpath ~/Dropbox/MatlabCode/Repos/sNIMclass/
        addpath(genpath('~/Dropbox/MatlabCode/Repos/L1General/'))    
        addpath(genpath('~/Dropbox/MatlabCode/Repos/minFunc_2012/'))  
end

%% load data
sessId = 8;
[Exp, S] = io.dataFactory(sessId);
eyePos = io.getCorrectedEyePos(Exp, 'usebilinear', true);
S.rect = [-10 0 30 40];

Exp.vpx.smo(:,2:3) = eyePos;

%% regenerate stimulus with current eye position
regenerate = true;
if regenerate
    options = {'stimulus', 'Gabor', ...
        'testmode', false, ...
        'eyesmooth', 9, ... % bins
        't_downsample', 2, ...
        's_downsample', 1, ...
        'includeProbe', true, ...
        'correctEyePos', false, ...
        'binsize', 2};
    
    fname = io.dataGenerate(Exp, S, options{:});
end

% rename stimulus file
fname = strrep(Exp.FileTag, '.mat', '_Gabor.mat');
dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
fprintf('Loading and renaming file\n')
tmp = load(fullfile(dataDir, fname)); %#ok<NASGU>
fname = strrep(Exp.FileTag, '.mat', '_Gabor_pre.mat');
save(fullfile(dataDir, fname), '-v7', '-struct', 'tmp')
clear tmp
fprintf('Done\n')
%% load
fprintf('Loading stimulus\n')
fname = strrep(Exp.FileTag, '.mat', '_Gabor_pre.mat');
load(fullfile(dataDir, fname));
fprintf('Done\n')


%% Build design matrix for STA
eyePosAtFrame = eyeAtFrame - mean(eyeAtFrame);

NY = size(stim,2)/NX;
nlags = ceil(.05/dt);
NC = size(Robs,2);

dims = [NX NY];

fprintf('Building time-embedded stimulus\n')
tic
spar = NIM.create_stim_params([nlags dims], 'tent_spacing', 2);
X = NIM.create_time_embedding(stim, spar);
X = X-mean(X);
X = zscore(X);
fprintf('Done [%02.2f]\n', toc)

%% run STA
ix = valdata == 1 & labels == 1 & (hypot(eyePosAtFrame(:,1), eyePosAtFrame(:,2)) < 1*Exp.S.pixPerDeg);
disp(sum(ix))

f = @(x) abs(x);
% C = f(X)'*f(X);
fprintf('Running STA...')
xy = [f(X(ix,:)) ones(sum(ix),1)]'*Robs(ix,:);
fprintf('[%02.2f]\n', toc)
xy = xy(1:end-1,:);

% d = size(X,2);
% stas = (C + 10e5*eye(d))\xy;
stas = xy;

% plot all
figure(10); clf
for cc = 1:NC
    a = reshape(stas(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')
    imagesc(a)
%     plot(a)
end

figure(11); clf
for cc = 1:NC
    a = reshape(xy(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')
    imagesc(xax, yax, reshape(a(3,:), dims), [0 1])
    colormap gray
end

%% Try to correct the calibration

bins = -10:.5:10;
ppd = Exp.S.pixPerDeg;
[xx,yy] = meshgrid(bins*ppd, bins*ppd);
xx = xx(:);
yy = yy(:);
n = numel(xx);
rfxys = nan(n,2);
figure; clf

% get global RF (eye in center)
% ix = valdata == 1 & labels == 1;
ix = valdata == 1 & labels == 1 & (hypot(eyePosAtFrame(:,1), eyePosAtFrame(:,2)) < 1*Exp.S.pixPerDeg);
disp(sum(ix))

f = @(x) abs(x);
fprintf('Get global STA...')
xy = [f(X(ix,:)) ones(sum(ix),1)]'*mean(Robs(ix,:),2);
fprintf('[%02.2f]\n', toc)
xy = xy(1:end-1,:);
a = reshape(xy,[nlags prod(dims)]);  
a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
rf =reshape(a(3,:), dims).^2;    
imagesc(xax, yax, rf, [0 1]); hold on
rfxy0 = [0 0];
[rfxy0(1),rfxy0(2)] = radialcenter(rf);
rfxy0(1) = interp1(1:numel(xax), xax, rfxy0(1));
rfxy0(2) = interp1(1:numel(yax), yax, rfxy0(2));

plot(rfxy0(1), rfxy0(2), 'or')
num = nan(n,1);
drawnow

%%
for iBin = 1:n
    ix = valdata == 1 & labels == 1 & (hypot(eyePosAtFrame(:,1)-xx(iBin), eyePosAtFrame(:,2)-yy(iBin)) < 50);
    num(iBin) = sum(ix);
    fprintf('%d/%d\n', iBin, n)
    xy = [f(X(ix,:)) ones(sum(ix),1)]'*sum(Robs(ix,:),2);
    xy = xy(1:end-1,:);
    a = reshape(xy,[nlags prod(dims)]);
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    rf =reshape(a(3,:), dims).^2;
    [tmpx,tmpy] = radialcenter(rf);
    rfxys(iBin,1) = interp1(1:numel(xax), xax, tmpx);
    rfxys(iBin,2) = interp1(1:numel(yax), yax, tmpy);
end
%%

dx = rfxys(:,1)-rfxy0(1);
dy = rfxys(:,2)-rfxy0(2);

figure; clf
thresh = 200; %at leas this many trials
iix = num>thresh;
quiver(xx(iix),yy(num>thresh),dx(num>thresh),dy(num>thresh))
xlabel('X position (pixels)')
ylabel('Y position (pixels)')
title('Error Map')
%%

Xgrid = [xx(iix) yy(iix)]/Exp.S.pixPerDeg;

Ygrid = [xx(iix) + dx(iix), yy(iix) + dy(iix)]/Exp.S.pixPerDeg;
figure(2); clf
plot(Xgrid(:,1), Xgrid(:,2), '+k')
hold on
plot(Ygrid(:,1), Ygrid(:,2), 'o')
xlabel('X position (d.v.a.)')
ylabel('Y position (d.v.a.)')
title('Error Map Degrees')

%% learn a function to correct eye position

ur = Xgrid(:,1) >= 0 & Xgrid(:,2) > 0;
ul = Xgrid(:,1) <= 0 & Xgrid(:,2) > 0;
lr = Xgrid(:,1) >= 0 & Xgrid(:,2) < 0;
ll = Xgrid(:,1) <= 0 & Xgrid(:,2) < 0;

indices = {ur, ul, lr, ll};
yhat = zeros(size(Ygrid));
w = cell(4, 1);
for i = 1:4
    iix = indices{i};
    Xd = [Ygrid(iix,:) Ygrid(iix,:).^2 Ygrid(iix,:).^3];

    w{i} = (Xd'*Xd)\(Xd'*Xgrid(iix,:));

    yhat(iix,:) = Xd*w{i};
end


cmap = lines;
plot(yhat(:,1), yhat(:,2), '.', 'Color', cmap(2,:))

%%

indices = {eyePos(:,1) >= 0 & eyePos(:,2) > 0, ...
    eyePos(:,1) <= 0 & eyePos(:,2) > 0, ...
    eyePos(:,1) >= 0 & eyePos(:,2) < 0, ...
    eyePos(:,1) <= 0 & eyePos(:,2) < 0};

eyePos2 = eyePos;
for i = 1:4
    iix = indices{i};
    Xd = [eyePos(iix,:) eyePos(iix,:).^2 eyePos(iix,:).^3];
    eyePos2(iix,:) = Xd*w{i};
end

%%
%%
% functions to correct
% Fx = scatteredInterpolant(Ygrid(:,1),Ygrid(:,2),Xgrid(:,1));
% Fy = scatteredInterpolant(Ygrid(:,1),Ygrid(:,2),Xgrid(:,2));



figure(2); clf
plot(Xgrid(:,1), Xgrid(:,2), '+k')
hold on
% plot(Y(:,1), Y(:,2), 'o')


%% test it

bins = -16:.5:16;
ppd = Exp.S.pixPerDeg;
[xx,yy] = meshgrid(bins*ppd, bins*ppd);
xx = xx(:);
yy = yy(:);

xhat = Fx(xx/ppd, yy/ppd);
yhat = Fy(xx/ppd, yy/ppd);
plot(xhat, yhat, '.')



































%%
%%

% S.rect = [-10 0 30 40];

eyePos = io.getCorrectedEyePos(Exp, 'usebilinear', true);
eyeX = Fx(eyePos(:,1), eyePos(:,2));
eyeY = Fy(eyePos(:,1), eyePos(:,2));
%% regenerate data with new parameters?
Exp.vpx.smo(:,2) = eyeX;
Exp.vpx.smo(:,3) = eyeY;
%%
regenerate = true;
if regenerate
options = {'stimulus', 'Gabor', ...
    'testmode', 50, ...
    'eyesmooth', 9, ... % bins
    't_downsample', 2, ...
    's_downsample', 1, ...
    'includeProbe', true, ...
    'correctEyePos', false, ...
    'binsize', 1};

fname = io.dataGenerate(Exp, S, options{:});
end

%% load stimulus

fname = strrep(Exp.FileTag, '.mat', '_Gabor.mat');
dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
fprintf('Loading stimulus\n')
load(fullfile(dataDir, fname))
fprintf('Done\n')


%%
eyePosAtFrame = eyeAtFrame - mean(eyeAtFrame);

NY = size(stim,2)/NX;
nlags = ceil(.05/dt);
NC = size(Robs,2);
%% Test RFs?
dims = [NX NY];

fprintf('Building time-embedded stimulus\n')
tic
spar = NIM.create_stim_params([nlags dims], 'tent_spacing', 2);
X = NIM.create_time_embedding(stim, spar);
X = X-mean(X);
X = zscore(X);
fprintf('Done [%02.2f]\n', toc)


%%
% ix = valdata == 1 & labels == 1;
ix = valdata == 1 & labels == 1 & (hypot(eyePosAtFrame(:,1), eyePosAtFrame(:,2)) < 5*Exp.S.pixPerDeg);
disp(sum(ix))
NTall = size(X,1);
cids = [1 3 4 9 13 18:23 26:28 34:NC];
% cids = 28;
N = numel(cids);
[sacav, ~, saclags] = eventTriggeredAverage(Robs, slist(:,2), [-1 1]*ceil(.2/dt));

figure; clf
plot(saclags, sacav/dt)

saclags = [3];
nsaclags = numel(saclags);
stassac = cell(nsaclags,1);
d = size(X,2);
for i = 1:nsaclags
%     inds = intersect(find(valdata), unique(bsxfun(@plus, slist(:,2), saclags(i)+(0:10)))); 
%     inds = unique(slist(:,2) + saclags(i) + (0:2);
    inds = find(ix);
    fprintf('lag: %d, %d valid samples\n', i, numel(inds))
    xy = [(X(inds,:)) ones(numel(inds),1)]'*Robs(inds,cids);
%     stasac{i} = (C + 10e6*eye(d))\xy(1:end-1,:);
    stasac{i} = xy(1:end-1,:);
    figure; clf
    for cc = 1:N
        a = reshape(stasac{i}(:,cc), [nlags prod(dims)]);
        a = (a - min(a(:))) / (max(a(:)) - min(a(:)));
        for ilag = 1:nlags
            subplot(N, nlags, (cc-1)*nlags + ilag, 'align')
            imagesc(xax, yax, reshape(a(ilag,:), [NX NY]), [0 1])
            axis off
            drawnow
        end
    end
%         subplot(6,6,cc, 'align')
%         imagesc(a)
%         axis off
    colormap(parula)
%     end
    drawnow
end

 


%%
f = @(x) (x);
C = f(X)'*f(X);
fprintf('Running STA...')
xy = [f(X(ix,:)) ones(sum(ix),1)]'*Robs(ix,:);
fprintf('[%02.2f]\n', toc)
xy = xy(1:end-1,:);



%%
d = size(X,2);
stas = (C + 10e5*eye(d))\xy;
stas = xy;

%% plot all
figure(10); clf
for cc = 1:NC
    a = reshape(stas(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')
%     imagesc(a)
    plot(a)
end

%%
figure; clf
for cc = 1:NC
    a = reshape(xy(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')
    imagesc(xax, yax, reshape(a(3,:), dims), [0 1])
    
end

%%

[xx,yy] = meshgrid(-200:25:200, -200:25:200);
xx = xx(:);
yy = yy(:);
n = numel(xx);
rfxys = nan(n,2);
figure; clf

% get global
ix = valdata == 1 & labels == 1;

f = @(x) abs(x);
fprintf('Running STA...')
xy = [f(X(ix,:)) ones(sum(ix),1)]'*sum(Robs(ix,:),2);
fprintf('[%02.2f]\n', toc)
xy = xy(1:end-1,:);
a = reshape(xy,[nlags prod(dims)]);  
a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
rf =reshape(a(3,:), dims).^2;    
imagesc(rf, [0 1]); hold on
rfxy0 = [0 0];
[rfxy0(1),rfxy0(2)] = radialcenter(rf);
plot(rfxy0(1), rfxy0(2), 'or')
num = nan(n,1);
tic
for iBin = 1:n
    ix = valdata == 1 & labels == 1 & (hypot(eyePosAtFrame(:,1)-xx(iBin), eyePosAtFrame(:,2)-yy(iBin)) < 50);
    num(iBin) = sum(ix);
    disp(iBin)
    xy = [f(X(ix,:)) ones(sum(ix),1)]'*sum(Robs(ix,:),2);
    fprintf('[%02.2f]\n', toc)
    xy = xy(1:end-1,:);
    a = reshape(xy,[nlags prod(dims)]);
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    rf =reshape(a(3,:), dims).^2;
    [rfxys(iBin,1),rfxys(iBin,2)] = radialcenter(rf);
end

%%
dx = rfxys(:,1)-rfxy0(1);
dy = rfxys(:,2)-rfxy0(2);

figure; clf
thresh = 500;
iix = num>thresh;
quiver(xx(iix),yy(num>thresh),dx(num>thresh),dy(num>thresh))



X = [xx(iix) yy(iix)]/Exp.S.pixPerDeg;
Ygrid = [xx(iix) + dx(iix), yy(iix) + dy(iix)]/Exp.S.pixPerDeg;
X = [X X.^2 X.^3 X.^4 ones(sum(iix),1)];
plot(X(:,1), X(:,2), '+k')
hold on
plot(Ygrid(:,1), Ygrid(:,2), 'o')

%%
w = (X'*X)\(X'*Ygrid);
yhat = X*w;
plot(yhat(:,1), yhat(:,2), '+r')

Fx = scatteredInterpolant(Ygrid(:,1),Ygrid(:,2),X(:,1));
Fy = scatteredInterpolant(Ygrid(:,1),Ygrid(:,2),X(:,2));

plot(X(:,1), X(:,2), '+k')
hold on
plot(Ygrid(:,1), Ygrid(:,2), 'o')

xhat = Fx(Ygrid(:,1), Ygrid(:,2));
yhat = Fy(Ygrid(:,1), Ygrid(:,2));
plot(xhat, yhat, '.')


% ep2 = F(eyePosAtFrame(:,1), eyePosAtFrame(:,2));
%%
figure; clf
for cc = 1:NC
    a = reshape(xy(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')
    imagesc(xax, yax, reshape(a(3,:), dims), [0 1])
    
end
%% plot one by one
for cc = 1:NC
    a = reshape(stas(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    figure(1); 
    for i = 1:nlags
        subplot(1, nlags, i, 'align')
        imagesc(xax, yax, reshape(a(i,:), dims), [0 1])
%         imagesc(xax/Exp.S.pixPerDeg/2, yax/Exp.S.pixPerDeg/2, reshape(a(i,:), dims), [0 1])
    end
%     subplot(2,1,1); hold off; plot(a,'b')
    
    title(sprintf('cell %d', cc ))
    drawnow
%     subplot(2,1,2); imagesc(xax/Exp.S.pixPerDeg/2, yax/Exp.S.pixPerDeg/2, reshape(a(3,:), dims))
    input('more?')
end


%%


% params = NIM.create_stim_params([nlags, dims(1), dims(2)], ...
%     'stim_dt', dt, ...
%     'upsampling', 1, ...
%     'tent_spacing', 2);

%% Loop over neurons, fit
NL_types = {'lin', 'rectlin'}; % define subunit as linear (note requires cell array of strings)
subunit_signs = [1, -1]; % determines whether input is exc or sup (mult by +1 in the linear case)
x_targs = [1, 1];
% Set initial regularization as second temporal derivative of filter
lambda_d2t = [0 0 0];
lambda_d2x = [.001 0 0];
lambda_l1 = [0 0.0001];
            
f = @(x) x;
NT = length(Robs);

ix = valdata == 1 & labels == 1 & (hypot(eyePosAtFrame(:,1), eyePosAtFrame(:,2)) < 500);% & probeDist > 50;
% ix = valdata == 1 & labels == 1 & eyePosAtFrame(:,1) > 0 & eyePosAtFrame(:,2) < 0;


train_inds0 = find(ix);
train_inds = randsample(train_inds0, ceil(.8*numel(train_inds0)));
test_inds = setdiff(train_inds0,train_inds);

nsubs = 1;
LN0 = NIM( spar, NL_types(1:nsubs), subunit_signs(1:nsubs), ...
    'xtargets', x_targs(1:nsubs), ...
    'd2t', lambda_d2t(1:nsubs), ...
    'l1', lambda_l1(1:nsubs), ...
    'd2x', lambda_d2x(1:nsubs));


%% fit filters
f = @(x) (x);

cc = 4; %cids(4);
LN0.subunits(1).filtK = stas(:,cc);
if nsubs > 1
LN0.subunits(2).filtK = stas(:,cc);
end
LN = LN0.fit_filters(Robs(:,cc), {f(X)}, train_inds);
LN.display_model
drawnow

% LN = LN.reg_path( Robs(:,cc), f(X), train_inds, test_inds, 'lambdaID', 'd2x' );
% LN.display_model


%%

LN = LN.reg_path( Robs(:,cc), X, train_inds, test_inds, 'lambdaID', 'l1' );
LN.display_model
drawnow

% LN.set_reg_params('l1', 10)
% LN = LN0.fit_filters(Robs(:,cc), {f(X)}, train_inds);
% LN.display_model
%%
NIM0 = LN.add_subunits( {'rectlin'}, -1);
NIM0 = NIM0.fit_filters(Robs(:,cc), {f(X)}, train_inds);
NIM0.display_model


%%
sLN = sNIM(LN, 1, spar);
sLN.subunits.reg_lambdas.d2x = 10e4;
sLN.subunits.reg_lambdas.d2t = 10e2;
sLN.subunits.reg_lambdas.l1 = 100;
sLN = sLN.fit_filters(Robs(:,cc), {stim}, train_inds)
sLN.display_model

sLN.add_subunits({'quad'}, 1)
sLN = sLN.fit_filters(Robs(:,cc), {stim})
sLN.display_model


%% GLM: add spike-history term and refit
GLM0 = LN.init_spkhist( 20, 'doubling_time', 5 ); 
GLM0 = GLM0.fit_filters( Robs(:,cc), X, find(ix), 'silent', 0 ); % silent suppresses optimization output

% Display model components
GLM0.display_model('Xstims',X,'Robs',Robs(:,cc))

%%
% Compare likelihoods
[LN.eval_model( Robs(:,cc), X, test_inds )  GLM0.eval_model( Robs(:,cc), X, test_inds )]
% Also available in fit-structures
[LN.fit_props.LL GLM0.fit_props.LL]
	
%% NIM: linear plus suppressive; like Butts et al (2011)
% Add an inhibitory input (with delayed copy of GLM filter and fit 
delayed_filt = NIM.shift_mat_zpad( LN.subunits(1).filtK, 4 );
NIM0 = GLM0.add_subunits( {'rectlin'}, -1, 'init_filts', {delayed_filt} );
NIM0 = NIM0.fit_filters( Robs(:,cc), X, find(ix) );

% Allow threshold of suppressive term to vary
NIM1 = NIM0.fit_filters( Robs(:,cc), X, 'fit_offsets', 1 ); % doesnt make huge difference
% Search for optimal regularization
NIM1 = NIM1.reg_path( Robs(:,c), X, find(ix), test_inds, 'lambdaID', 'd2t' );

% Compare subunit filters (note delay of inhibition)
NIM1.display_subunit_filters()

