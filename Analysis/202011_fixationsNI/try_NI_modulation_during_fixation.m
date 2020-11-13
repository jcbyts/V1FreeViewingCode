
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

addpath Analysis/202001_K99figs_01  
addpath Analysis/manuscript_freeviewingmethods/

%% load data
sessId = 5;
sorter = 'jrclustwf';
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', sorter);

%%
srf = spatial_RF_single_session(Exp);

%%

grf = grating_RF_single_session(Exp);

%%
win = [-.1 .5]; % centered on fixation start
binsize = 1e-3; % raster resolution
eyePosInterpolationMethod = 'linear';

stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

% fixation times
fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));

bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
fixon(bad) = [];
sacon(bad) = [];

[valid, epoch] = getTimeIdx(fixon, tstart, tstop);
fixon = fixon(valid);
sacon = sacon(valid);
fixTrial = epoch(valid);

fixdur = sacon - fixon;
[~, ind] = sort(fixdur);

% --- eye position
eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1)); % time
remove = find(diff(eyeTime)==0); % bad samples

% filter eye position with 3rd order savitzy-golay filter
eyeX = sgolayfilt(Exp.vpx.smo(:,2), 3, 9); % smooth (preserving tremor)
eyeX(isnan(eyeX)) = 0;
eyeY = sgolayfilt(Exp.vpx.smo(:,3), 3, 9);
eyeY(isnan(eyeY)) = 0;

% remove bad samples
eyeTime(remove) = [];
eyeX(remove) = [];
eyeY(remove) = [];

lags = win(1):binsize:win(2);


nlags = numel(lags);
nfix = numel(fixon);
cids = Exp.osp.cids;
NC = numel(cids);

spks = zeros(nfix,NC,nlags);

xpos = zeros(nfix, 2); % pre and post saccade
ypos = zeros(nfix, 2);

[~, ~, id1] = histcounts(fixon, eyeTime);
[~, ~, id2] = histcounts(sacon, eyeTime);

for ifix = 1:nfix
   prefix = id1(ifix) + (-100:-50);
   postfix = id1(ifix):(id2(ifix)-20);
   xpos(ifix,1) = mean(eyeX(prefix));
   xpos(ifix,2) = mean(eyeX(postfix));
   ypos(ifix,1) = mean(eyeY(prefix));
   ypos(ifix,2) = mean(eyeY(postfix));
end

disp('Binning Spikes')
st = Exp.osp.st;
clu = Exp.osp.clu;
keep = ismember(clu, cids);
st = st(keep);
clu = double(clu(keep));

st0 = min(st);

bs = (st==0) + ceil(st/binsize);
spbn = sparse(bs, clu, ones(numel(bs), 1));
spbn = spbn(:,cids);
blags = ceil(lags/binsize);
bfixon = ceil(fixon/binsize);

% Do the binning here
for i = 1:nlags
    spks(:,:,i) = spbn(bfixon + blags(i),:);
end

disp('Done')

%%

cc = cc+1;
if cc > NC
    cc = 1;
end
[i,j] = find(squeeze(spks(ind,cc,:)));
figure(1); clf
subplot(1,2,1)
plot.raster(j,i,10)
axis tight
title(cc)

subplot(2,2,2)
plot(srf.fine(cc).sta)

subplot(2,2,4)
plot(grf(cc).sta)

%%

rfLocations = nan(NC,2);
for cc = 1:NC
    if ~isempty(srf.fine(cc).gfit)
        rfLocations(cc,:) = srf.fine(cc).gfit.mu;
    end
end

figure(1); clf
plot(rfLocations(:,1), rfLocations(:,2), 'o')
nanmedian(rfLocations)
% % rfCenter = srf.fine(cc).gfit.mu;
% rfCenter = [.2 -.8];
rfCenter = [1.5 -1.2];
cc = 25;
%%

rect = [-1 -1 1 1]*40; % window centered on RF
ppd = Exp.S.pixPerDeg;
ctr = Exp.S.centerPix;

dims = [rect(4)-rect(2) rect(3)-rect(1)];
fixIms = zeros(dims(1), dims(2), nfix, 2);
fftIms = zeros(dims(1), dims(2), nfix, 2);

hwin = hanning(dims(1))*hanning(dims(2))';
%%
for iTrial = 1:numel(validTrials)
    
    fprintf('%d/%d\n', iTrial, numel(validTrials))
    
    thisTrial = validTrials(iTrial);
    
    % load image
    try
        Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));
    catch
        try
            Im = imread(fullfile(fileparts(which('marmoV5')), strrep(Exp.D{thisTrial}.PR.imageFile, '\', filesep)));
        catch
            fprintf(1, 'regenerateStimulus: failed to load image\n')
            continue
        end
    end
    
    % zero mean
    Im = mean(Im,3)-127;
    Im = imresize(Im, fliplr(Exp.S.screenRect(3:4)));
    
    % loop over fixations
    fixtrial = find(validTrials(fixTrial) == validTrials(iTrial));
    nft = numel(fixtrial);
    
    figure(1); clf
    imagesc(Im); hold on
    colormap gray
    plot(xpos(fixtrial,:)*ppd + ctr(1), -ypos(fixtrial,:)*ppd + ctr(2), 'ro')
    drawnow
    
    
    for ifix = 1:nft
        
        thisfix = fixtrial(ifix);
        eyeX = xpos(thisfix, 2)*ppd + ctr(1) + rfCenter(1)*ppd;
        eyeY = -ypos(thisfix, 2)*ppd + ctr(2) + rfCenter(2)*ppd;
        
        % center on eye position
        tmprect = rect + [eyeX eyeY eyeX eyeY];
        
        imrect = [tmprect(1:2) (tmprect(3)-tmprect(1))-1 (tmprect(4)-tmprect(2))-1];
        I = imcrop(Im, imrect); % requires the imaging processing toolbox
        
%         figure(2); clf
%         subplot(131)
%         imagesc(Im); hold on; colormap gray
%         plot(eyeX, eyeY, 'or')
%         plot([imrect(1) imrect(1) + imrect(3)], imrect([2 2]), 'r')
%         plot([imrect(1) imrect(1) + imrect(3)], imrect(2)+imrect([4 4]), 'r')
%         plot(imrect([1 1]),[imrect(2), imrect(2) + imrect(4)], 'r')
%         plot(imrect(1)+imrect([3 3]), [imrect(2), imrect(2) + imrect(4)], 'r')
%         subplot(132)
        
        if ~all(size(I)==dims(1))
            continue
        end
        
        Iwin = (I - mean(I(:))).*hwin;
%         imagesc(Iwin)
        

        fIm = fftshift(fft2(Iwin));
        xax = -dims(1)/2:dims/2;
        xax = xax / dims(1) * ppd;
        
%         subplot(133)
%         imagesc(xax, xax, abs(fIm))
%         drawnow
        
        fixIms(:,:,thisfix,2) = Iwin;
        fftIms(:,:,thisfix,2) = fIm;
        
    end
    
    
end


%%
cc = cc+1;
if cc > NC
    cc = 1;
end
[kx, ky] = meshgrid(xax(2:end), xax(2:end));
[ori, cpd] = cart2pol(kx, ky);     
ori = wrapToPi(ori);

params = [grf(cc).rffit.oriBandwidth/180*pi, ...
    grf(cc).rffit.oriPref/180*pi, ...
    grf(cc).rffit.sfPref,...
    grf(cc).rffit.sfBandwidth,...
    grf(cc).rffit.amp, ...
    grf(cc).rffit.base];

Ifit = prf.parametric_rf(params, [ori(:), cpd(:)]);

rf = reshape(Ifit, size(ori));
figure(1); clf
set(gcf, 'Color', 'w')
subplot(2,2,1)
imagesc(xax, xax, rf)
title('RF')
xlabel('sf')
ylabel('sf')

freshape = abs(reshape(fftIms(:,:,:,2), [prod(dims) nfix]));

fproj = freshape'*rf(:);

good_trials= find(sum(freshape)~=0);

subplot(2,2,2)
histogram(fproj(good_trials), 'FaceColor', .5*[1 1 1])


levels = prctile(fproj(good_trials), [10 90]);

lowix = good_trials(fproj(good_trials) < levels(1));
hiix = good_trials(fproj(good_trials) > levels(2));

hold on
plot(levels(1)*[1 1], ylim, 'r', 'Linewidth', 2)
plot(levels(2)*[1 1], ylim, 'b', 'Linewidth', 2)

xlabel('Generator Signal')
ylabel('Fixation Count')

% psth
subplot(2,2,3)

sm = 20;
spksfilt = filter(ones(sm,1), 1, squeeze(spks(:,cc,:))')';

psthhi = mean(spksfilt(hiix,:))/binsize/sm;
psthlow = mean(spksfilt(lowix, :))/binsize/sm;

pstvhi = std(spksfilt(hiix,:))/binsize/sm;
pstvlow = std(spksfilt(lowix, :))/binsize/sm;

plot.errorbarFill(lags, psthhi, pstvhi/sqrt(numel(hiix)), 'b', 'FaceColor', 'b'); hold on
plot.errorbarFill(lags, psthlow, pstvlow/sqrt(numel(lowix)), 'r', 'FaceColor', 'r')
axis tight
xlim([-.05 .25])
title(cc)
xlabel('Time from fix on (s)')
ylabel('Firing Rate')

subplot(2,2,4)
plot(lags, (psthhi-psthlow) ./ sqrt( (pstvhi.^2 + pstvlow.^2)/2) , 'k'); hold on
% plot(xlim, [1 1], 'k--')
xlim([-.05 .25])
xlabel('Time from fix on (s)')
ylabel('Ratio (pref / null)')


[i,j] = find(squeeze(spks(ind,cc,:)));
figure(2); clf
subplot(1,2,1)
plot.raster(j,i,10)
axis tight
title(cc)

subplot(2,2,2)
plot(srf.fine(cc).sta)
title('spatial sta')
xlabel('time lags')

subplot(2,2,4)
plot(grf(cc).sta)
title('hartley sta')
xlabel('time lags')
%%
