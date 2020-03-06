function [eyePos2,eyePos3] = getCorrectedEyeposRF(Exp, S, eyePos)

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
figure; clf
for cc = 1:NC
    a = reshape(stas(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')
    imagesc(a)
%     plot(a)
end

figure; clf
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
    ix = valdata == 1 & labels == 1 & (hypot(eyePosAtFrame(:,1)-xx(iBin), eyePosAtFrame(:,2)-yy(iBin)) < 20);
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
figure; clf
plot(Xgrid(:,1), Xgrid(:,2), '+k')
hold on
plot(Ygrid(:,1), Ygrid(:,2), 'o')
xlabel('X position (d.v.a.)')
ylabel('Y position (d.v.a.)')
title('Error Map Degrees')

%% learn a function to correct eye position

 rbfOpts = {'multiquadric', 'RBFConstant', 1, 'RBFSmooth', 10};
 opX = rbfcreate(Ygrid', Xgrid(:,1)', 'RBFFunction', rbfOpts{:});
    rbfcheck(opX);
    opY = rbfcreate(Ygrid', Xgrid(:,2)', 'RBFFunction', rbfOpts{:});
    rbfcheck(opY);

    % only fix the central points
%     iix = hypot(eyePos(:,1), eyePos(:,2)) < 5*ppd;
    
    
    

% only fix the central pointe
iixx = eyePos(:,1) > -4*ppd & eyePos(:,1) < 4*ppd;
iiyy = eyePos(:,2) > -4*ppd & eyePos(:,2) < 4*ppd;

eyePos2 = eyePos;
iix = iixx & iiyy;
    eyePos2(iix,1) = rbfinterp(eyePos(iix,:)', opX)';
    eyePos2(iix,2) = rbfinterp(eyePos(iix,:)', opY)';    
% eyePos2(iix, 1) = Fx(eyePos(iix,1), eyePos(iix,2));
% eyePos2(iix, 2) = Fy(eyePos(iix,1), eyePos(iix,2));


if nargout > 1
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

eyePos3 = eyePos;
for i = 1:4
    iix = indices{i};
    Xd = [eyePos(iix,:) eyePos(iix,:).^2 eyePos(iix,:).^3];
    eyePos3(iix,:) = Xd*w{i};
end



iix = hypot(eyePos2(:,1), eyePos2(:,2)) > 10;
eyePos3(iix,:) = eyePos(iix,:);
end