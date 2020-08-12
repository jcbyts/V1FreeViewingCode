
%%

tmp = load(fullfile('Data', 'Cstim3_raw_output.mat'));
tmp2 = load(fullfile('Data', 'modelSTAs.mat'));


%%

figure(1); clf
dim = [40 40];
NC = size(tmp.stas, 3);


for cc = 1:NC
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')

    a = squeeze(tmp.stas(:,:,(cc)));
    a = (a - median(a(:))) / std(a(:));
    [~, peakLag] = max(mean(a));
    
    
       
    I = reshape(a(:,peakLag), dim);
   
    imagesc(I)
    axis off
    
end

figure(2); clf
dim = [40 40];
NC = size(tmp.stas, 3);


for cc = 1:NC
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')

    a = squeeze(tmp2.staModel(:,:,(cc)));
    a = (a - median(a(:))) / std(a(:));
    [~, peakLag] = max(mean(a));
    
    
       
    I = reshape(a(:,peakLag), dim);
   
    imagesc(I)
    axis off
    title(cc)
    
end
    


%%
dims = [40 40];
xax = ((-dims(2)/2 + 1):dims(2)/2)*1.5;
yax = -(1:dims(1))*1.5;
figure(1); clf
NC = size(tmp.stas,3);
cc = mod(cc + 1, NC); cc = max(cc, 1);
nlags = size(tmp.stas,2);
a = squeeze(tmp.stas(:,:,cc));
a = (a - median(a(:))) / std(a(:));

b = squeeze(tmp2.staModel(:,:,cc));
b = (b - median(b(:))) / std(b(:));

clim = max(abs(a(:)))*[-1 1]*.75;    
for ilag = 1:nlags
    subplot(2,nlags,ilag, 'align')
    imagesc(xax, yax, reshape(a(:,ilag), dims), clim)
    axis xy
    
    subplot(2,nlags,nlags + ilag, 'align')
    imagesc(xax, yax, reshape(b(:,ilag), dims), clim)
    axis xy
%     contourf(reshape(a(:,ilag), [40 40]),[-4:.1:-2  0 2:6], 'Linestyle', 'none')
end
title(cc)
colormap(gray)

%%

cmap = gray;
[xx,tt,yy] = meshgrid(xax, (1:nlags)*8, yax);
% [xx,yy] = meshgrid(xax, yax);
I = reshape(a, [dims nlags]);
I = permute(I, [3 2 1]);

I2 = reshape(b, [dims nlags]);
I2 = permute(I2, [3 2 1]);

figure(2); clf
set(gcf, 'Color', 'w')
subplot(2,1,1)
h = slice(xx,tt, yy, I, [], (1:10)*8,[]);
set(gca, 'CLim', [-9 9])
for i = 1:numel(h)
    h(i).EdgeColor = 'none';
%     text(-12, i*8-4, 5, sprintf('%i ms', (i)*8))
end
view(79,11)
colormap(cmap)
axis off

subplot(2,1,2)
h = slice(xx,tt, yy, I2, [], (1:10)*8,[]);
set(gca, 'CLim', [-9 9])
for i = 1:numel(h)
    h(i).EdgeColor = 'none';
%     text(-12, i*8-4, 5, sprintf('%i ms', (i)*8))
end
view(79,11)
colormap(cmap)
axis off

%%

hold on
plot3(10+[10 16], [1 1]*39, -[50 50], 'r', 'Linewidth', 2)
% plot3(10+[10 16], [1 1]*30.5, -[50 50], 'r', 'Linewidth', 2)
% text(40, 32, -50, '0.1 d.v.a', 'Color', 'r')

plot.fixfigure(gcf, 8, [14 3])
saveas(gcf, fullfile('Figures', 'K99', sprintf('sta%02.0f.png', cc)))
%% add paths

user = 'jakework';
addFreeViewingPaths(user);

switch user
    case 'jakework'
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\NIMclass
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\sNIMclass
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\L1General'))    
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\minFunc_2012'))  
%         addpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\lowrankrgc')
%         setpaths_lowrankRGC
    case 'jakelaptop'
        addpath ~/Dropbox/MatlabCode/Repos/NIMclass/
        addpath ~/Dropbox/MatlabCode/Repos/sNIMclass/
        addpath(genpath('~/Dropbox/MatlabCode/Repos/L1General/'))    
        addpath(genpath('~/Dropbox/MatlabCode/Repos/minFunc_2012/'))  
end

%% load data
sessId = 12;
[Exp, S] = io.dataFactory(sessId);
out = load(fullfile('Data','gmodsac0_output.mat'));
eyePos = io.getCorrectedEyePosFixCalib(Exp);
eyePos(:,1) = sgolayfilt(eyePos(:,1), 3, 9);
eyePos(:,2) = sgolayfilt(eyePos(:,2), 3, 9);
Exp.vpx.smo(:,2:3) = eyePos;

stimuli = {'Gabor', 'Grating'};
dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
%% load stimulus
stim = [];
Robs = [];
iix = [];
eyeiix = [];
spacerS = zeros(60, 1600);
spacerR = zeros(60, 61);
for i = 1:numel(stimuli)
    stimulus = stimuli{i};
    fname = strrep(Exp.FileTag, '.mat', ['_' stimulus '.mat']);
    
    fprintf('Loading [%s] stimulus\n', stimulus)
    tmp = load(fullfile(dataDir, fname));
    
    fprintf('Done\n')
    eyePosAtFrame = tmp.eyeAtFrame - Exp.S.centerPix;
    eyeiix = [eyeiix; eyePosAtFrame];
%     eyeix = eyePosAtFrame(:,1) > 0 & eyePosAtFrame(:,2) > 0 & ...
%          eyePosAtFrame(:,1) < 200 & eyePosAtFrame(:,2) < 200;
%     eyeix = hypot(eyePosAtFrame(:,1),eyePosAtFrame(:,2)) < 200;
%     ix = tmp.valdata == 1 & tmp.labels == 1 & eyeix;
    ix = tmp.valdata == 1 & tmp.labels == 1;
    if i>1
        iix = [iix; false(60,1); ix(:)];
        stim = [stim; spacerS; tmp.stim];
    Robs = [Robs; spacerR; tmp.Robs];
    else
    iix = [iix; ix(:)];
    stim = [stim; tmp.stim];
    Robs = [Robs; tmp.Robs];
    end
end
cc = 1;
nky = numel(tmp.yax)/tmp.opts.s_downsample;
nkx = numel(tmp.xax)/tmp.opts.s_downsample;
xax = tmp.xax;
yax = tmp.yax;
clear tmp
%%
load(fullfile('Data', 'Cstim2.mat'))
stim = Cstim; 
clear Cstim
Robs(:,sum(Robs) < 500) = [];
Robs = out.Robs(1:size(Robs,1),:);
Rhat = out.Robs(1:size(Robs,1), :);

%%
nlags = 10; % number of time lags in STA
NC = size(Robs,2);

% Test RFs?
dims = [nkx nky];

fprintf('Building time-embedded stimulus\n')
tic
spar = NIM.create_stim_params([nlags dims]); %, 'tent_spacing', 1);
X = NIM.create_time_embedding(stim, spar);
X = X./std(X);
fprintf('Done [%02.2f]\n', toc)

fprintf('Detrending spike rates\n')
tic
sps = Robs;
for cc = 1:NC
    sps(:,cc) = detrend(sps(:,cc), 'linear');
end
fprintf('Done [%02.2f]\n', toc)

%% compute STA
inds = find(iix & (hypot(eyeiix(:,1), eyeiix(:,2)) < 150));
fprintf('%d valid samples\n', numel(inds))
xy = [(X(inds,:)) ones(numel(inds),1)]'*sps(inds,:);
% xy2 = [X(inds,:) ones(numel(inds), 1)]'*Rhat(inds,:);

    
%% plot
ppa = 60/Exp.S.pixPerDeg;
d = size(X,2);
% [~, cids] = sort(Exp.osp.clusterDepths);
cids = 1:NC;
stas = xy(1:end-1,cids);
% stas2 = xy2(1:end-1,cids);
% stas = xy;

figure; clf
for cc = 1:NC
    a = reshape(stas(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')
    imagesc(xax*ppa, yax*ppa, reshape(a(4,:), dims), [0 1])
    title(cc)
    
end

colormap gray

% figure; clf
% for cc = 1:NC
%     a = reshape(stas2(:,cc),[nlags prod(dims)]);  
%     a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
%     subplot(ceil(sqrt(NC)), round(sqrt(NC)), cc, 'align')
%     imagesc(xax*ppa, yax*ppa, reshape(a(4,:), dims), [0 1])
%     title(cc)
%     
% end


%% plot one by one
figure
for cc = 1:NC
    a = reshape(stas(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    
    b = reshape(stas2(:,cc),[nlags prod(dims)]);  
    b = (b - min(b(:))) ./ (max(b(:)) - min(b(:)));
    
    for i = 1:nlags
        subplot(2, nlags, i, 'align')
        imagesc(xax*ppa, yax*ppa, reshape(a(i,:), dims), [0 1])
        subplot(2,nlags,i+nlags,'align')
        imagesc(xax*ppa, yax*ppa, reshape(b(i,:), dims), [0 1])
%         xlim([0 25])
%         ylim([15 40])
%         imagesc(xax/Exp.S.pixPerDeg/2, yax/Exp.S.pixPerDeg/2, reshape(a(i,:), dims), [0 1])
    end
%     subplot(2,1,1); hold off; plot(a,'b')
    
    title(sprintf('cell %d', cc ))
    colormap gray
    drawnow
%     subplot(2,1,2); imagesc(xax/Exp.S.pixPerDeg/2, yax/Exp.S.pixPerDeg/2, reshape(a(3,:), dims))
    input('more?')
end

%%
figure(cc); clf
I = (sta - min(sta(:))) / (max(sta(:))-min(sta(:)));
for ilag = 1:nkt
    subplot(1,nkt,ilag, 'align')
    imagesc(reshape(I(ilag,:), [nky, nkx]), [0 1])
end
title(cc)

%%
[usta,ssta,vsta] = svd(sta); % svd to get low-rank approximation of stimulus

%% Plot the sta and low-rank approximation
subplot(321); imagesc(1:nkx,1:nkt,sta); % raw sta
title('STA'); xlabel('pixel'); ylabel('time lag');
subplot(323); plot(diag(ssta), 'o'); title('singular values of STA');
subplot(325); plot(usta(:,1:2)); title('top temporal components');
subplot(324); imagesc(reshape(vsta(:,1),nky,nkx)); title('1st spatial component');
subplot(326); imagesc(reshape(vsta(:,2),nky,nkx)); title('2nd spatial component');
subplot(322); imagesc(usta(:,1:2)*ssta(1:2,1:2)*vsta(:,1:2)'); title('rank-2 reconstruction');


%% Divide data into training and test
trainfrac = .8;
ntrain = ceil(trainfrac*size(stim,1));
stimtrain = stim(1:ntrain,:); stimtest = stim(ntrain+1:end,:);
spstrain = sps(1:ntrain); spstest = sps(ntrain+1:end); 

% function for computing test error (to be used later)
msefun = @(y)(mean((spstest-y).^2));  

%%  Estimate rank-1 W by coordinate ascent

fprintf('---------  Now fitting RANK 1 model ------ \n')
k_rank = 1; % rank of filter
lambda = 1; % ridge parameter
[what1,wt1,wx1] = bilinearRegress_coordAscent_xt(stimtrain,spstrain,nkt,k_rank,lambda);

% training error
msetr1 = mean((spstrain-sameconv(stimtrain*wx1,wt1)).^2);
Ypred1 = sameconv(stimtest*wx1,wt1);  % predicted Y, rank 1

%%  Run ASD on rank-1 spatial estimate

% Convert temporal vector to a unit vector
wt1nrm = norm(wt1);
wt1u = wt1/wt1nrm;

xstimtrain = sameconvcols(stimtrain,wt1u);

fprintf('---------  Now running ASD on rank-1 model ------ \n')
minlen = 1;
[wx1asd,asd1stats,dd1] = fastASD(xstimtrain,spstrain,[nky,nkx],minlen);

msetr1asd = mean((spstrain-sameconv(stimtrain*wx1asd,wt1)).^2); % training error
Ypred1asd = sameconv(stimtest*wx1asd,wt1u);  % predicted Y, rank 1
testMSE_rank1MLvsASD = [msefun(Ypred1), msefun(Ypred1asd)]

% Plot estimates so far
subplot(121); 
imagesc(reshape(wx1,nky,nkx)); axis image;
title('rank 1 MLE');
subplot(122); imagesc(reshape(wx1asd,nky,nkx)); axis image;
title('rank 1 ASD');
drawnow; 

%% Extract covariance straight-up MAP inference under learned ASD prior

% Construct inverse covariance matrix from learned ASD params
Bfft = kron(dd1.Bfft{2},dd1.Bfft{1});
cdiaginv = zeros(size(Bfft,2),1);
cdiaginv(dd1.ii) = diag(dd1.Cinv);
Cinv = diag(cdiaginv);
Cxinv = Bfft*Cinv*Bfft';
% Construct covariance matrix
cdiag = zeros(size(Bfft,2),1);
cdiag(dd1.ii) = 1./diag(dd1.Cinv);
Cxasd = Bfft*diag(cdiag)*Bfft'; 
imagesc(Cxasd)

% Make matrix for projecting out missing components
[UU,SS,VV] = svd(Cxasd);
ii = diag(SS)<SS(1)/1e12;
Bnull = UU(:,ii);

% Set prior inverse covariance for temporal params to identity 
Ctinv = speye(nkt);

%%  Re-estimate rank-1 W by MAP coordinate ascent

fprintf('---------  Now fitting RANK 1 model, MAP ------ \n')

k_rank = 1;
[what1map,wt1map,wx1map] = bilinearRegress_coordAscent_xtMAP(stimtrain,spstrain,nkt,k_rank,Ctinv,Cxinv);
wx1map = wx1map - Bnull*(Bnull'*wx1map); % project out component in null space of prior

% Compute training and test error
msetr1map = mean((spstrain-sameconv(stimtrain*wx1map,wt1map)).^2); % training error
Ypred1map = sameconv(stimtest*wx1map,wt1map);  % predicted Y, rank 1
testMSE = [msefun(Ypred1), msefun(Ypred1asd), msefun(Ypred1map)]

clf; imagesc(reshape(normalizecols([wx1, wx1asd, wx1map]),nky,[]));
drawnow;


%%  Estimate rank-2 W by MAP coordinate ascent

fprintf('---------  Now fitting RANK 2 model, MAP ------ \n')

k_rank = 2; % rank of filter
[what2map,wt2map,wx2map] = bilinearRegress_coordAscent_xtMAP(stimtrain,spstrain,nkt,k_rank,Ctinv,Cxinv);
wx2map = wx2map - Bnull*(Bnull'*wx2map); % project out component in null space of prior

% Compute training and test error
msetr2 = mean((spstrain-sameconv(stimtrain*wx2map,wt2map)).^2); % training error
Ypred2 = sameconv(stimtest*wx2map,wt2map);  % predicted Y, rank 2

% Report all errors and make fig
trainmse = [msetr1 msetr1asd msetr1map, msetr2]
testMSE = [msefun(Ypred1), msefun(Ypred1asd), msefun(Ypred1map) msefun(Ypred2)]

clf
imagesc(reshape(normalizecols([wx1, wx1asd, wx1map, wx2map]),nky,[]));
drawnow;


%%  Estimate rank-3 W by MAP coordinate ascent

fprintf('---------  Now fitting RANK 3 model, MAP ------ \n')

k_rank = 3; % rank of filter
[what3map,wt3map,wx3map] = bilinearRegress_coordAscent_xtMAP(stimtrain,spstrain,nkt,k_rank,Ctinv,Cxinv);
wx3map = wx3map - Bnull*(Bnull'*wx3map); % project out component in null space of prior

% Compute training and test error
msetr3 = mean((spstrain-sameconv(stimtrain*wx3map,wt3map)).^2); % training error
Ypred3 = sameconv(stimtest*wx3map,wt3map);  % predicted Y, rank 3

% Report all errors and make fig
trainmse = [msetr1 msetr1asd msetr1map, msetr2, msetr3]
testMSE = [msefun(Ypred1), msefun(Ypred1asd), msefun(Ypred1map) msefun(Ypred2) msefun(Ypred3)]

clf;
imagesc(reshape(normalizecols([wx1, wx1asd, wx1map, wx2map wx3map]),nky,[]));




