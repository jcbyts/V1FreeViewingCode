function fftrf = fixrate_by_fftrf(Exp, srf, grf, varargin)
% fftrf = fixrate_by_fftrf(Exp, srf, grf, varargin)

ip = inputParser();
ip.addParameter('plot', true)
ip.addParameter('win', [-.1 .5])
ip.addParameter('binsize', 1e-3)
ip.addParameter('smoothing', 20)
ip.addParameter('makeMovie', false)
ip.parse(varargin{:})



%% Bin spikes and eye position
sm = ip.Results.smoothing;
win = ip.Results.win; % centered on fixation start
binsize = ip.Results.binsize; % raster resolution

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
%
% cc = cc+1;
% if cc > NC
%     cc = 1;
% end
% [i,j] = find(squeeze(spks(ind,cc,:)));
% figure(1); clf
% subplot(1,2,1)
% plot.raster(lags(j),i,10)
% axis tight
% title(cc)
% xlabel('Time from Fixation Onset')
%
% subplot(2,2,2)
% plot(srf.fine(cc).sta)
% xlabel('Time lag')
% ylabel('ampltude')
% title('Spatial Mapping')
%
% subplot(2,2,4)
% plot(grf(cc).sta)
% xlabel('Time lag')
% ylabel('ampltude')
% title('Grating Mapping')

%% group units by RF location

rfLocations = nan(NC,2);
if isfield(srf, 'rfLocations')
    rfLocations = srf.rfLocations;
else
    for cc = 1:NC
        if ~isempty(srf.fine(cc).gfit)
            rfLocations(cc,:) = srf.fine(cc).gfit.mu;
        end
    end
end

figure(1); clf
plot(rfLocations(:,1), rfLocations(:,2), 'o'); hold on
rfCenter = nanmedian(rfLocations);
plot(rfCenter(1), rfCenter(2), 'ro')

xax = linspace(min(xlim), max(xlim), 100);
yax = linspace(min(ylim), max(ylim), 100);
[xx,yy] = meshgrid(xax, yax);

ix = ~any(isnan(rfLocations),2);
n = 3;
AIC = nan(n,1);
BIC = nan(n,1);
figure(2); clf
gmdists = cell(n,1);
for c = 1:n
    subplot(1,n,c)
    try
        evalc('gmdists{c} = fitgmdist(rfLocations(ix,:), c);');
        gmdist = gmdists{c};
        %     xx = linspace(min(xlim), max(xlim), 1000);
        %     plot(xx, gmdist.pdf(xx'))
        AIC(c) = gmdist.AIC;
        BIC(c) = gmdist.BIC;
        contourf(xx,yy,reshape(gmdist.pdf([xx(:) yy(:)]), size(xx)))
    end
    
end

[~, id] = min(BIC);
gmdist = gmdists{id};
% evalc('gmdist = fitgmdist(rfLocations(ix,:), id);');

% cluster and show RF clusters
figure(1); clf
clusts = cluster(gmdist, rfLocations);
mostCommon = mode(clusts);
clusts(isnan(clusts)) = mostCommon;
for c = 1:id
    ii = clusts==c;
    plot(rfLocations(ii,1), rfLocations(ii,2), 'o'); hold on
end

%% parameters of analysis

ppd = Exp.S.pixPerDeg;
ctr = Exp.S.centerPix;


% initialize output
stmp = [];
stmp.lags = [];
stmp.rfLocation = [];
stmp.rateHi = [];
stmp.rateLow = [];
stmp.stdHi = [];
stmp.stdLow = [];
stmp.nHi = [];
stmp.nLow = [];
stmp.rf.kx = [];
stmp.rf.ky = [];
stmp.rf.Ifit = [];
stmp.xproj.bins = [];
stmp.xproj.cnt = [];
stmp.cid = [];

fftrf = repmat(stmp, NC, 1);


% setup movie
exname = strrep(Exp.FileTag, '.mat', '');

for clustGroup = unique(clusts(:)')
    
    fig = figure(2);
    fig.Position = [100 100 800 250]; clf
    fig.Color = 'w';
    
    fname = sprintf('Figures/fftfixclip_%s_%d', exname, clustGroup);
    if ip.Results.makeMovie
        vidObj = VideoWriter(fname, 'MPEG-4');
        vidObj.FrameRate = 3;
        vidObj.Quality = 100;
        open(vidObj);
    end
    
    
    rfCenter = gmdist.mu(clustGroup,:);
    
    rfwidth = hypot(rfCenter(1), rfCenter(2));
    rfwidth = min(rfwidth, 2);
    rect = [-1 -1 1 1]*ceil(ppd*rfwidth); % window centered on RF
    dims = [rect(4)-rect(2) rect(3)-rect(1)];
    fixIms = zeros(dims(1), dims(2), nfix, 2);
    fftIms = zeros(dims(1), dims(2), nfix, 2);
    
    hwin = hanning(dims(1))*hanning(dims(2))';
    
    %% do fixation clipping and compute fft
    nTrials = numel(validTrials);
    if ip.Results.makeMovie
        nTrials = 4;
    end
    
    for iTrial = 1:nTrials
        
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
        if ip.Results.plot
            figure(1); clf
            imagesc(Im); hold on
            colormap gray
            plot(xpos(fixtrial,:)*ppd + ctr(1), -ypos(fixtrial,:)*ppd + ctr(2), 'ro')
            drawnow
        end
        
        % loop over fixations
        for ifix = 1:nft
            
            thisfix = fixtrial(ifix);
            eyeX = xpos(thisfix, 2)*ppd + ctr(1) + rfCenter(1)*ppd;
            eyeY = -ypos(thisfix, 2)*ppd + ctr(2) + rfCenter(2)*ppd;
            
            % center on eye position
            tmprect = rect + [eyeX eyeY eyeX eyeY];
            
            imrect = [tmprect(1:2) (tmprect(3)-tmprect(1))-1 (tmprect(4)-tmprect(2))-1];
            I = imcrop(Im, imrect); % requires the imaging processing toolbox
            
            if ip.Results.makeMovie
                figure(fig)
                subplot(131, 'align')
                imagesc(Im); hold on; colormap gray
                axis off
                plot(eyeX, eyeY, '.c', 'MarkerSize', 10)
                plot([imrect(1) imrect(1) + imrect(3)], imrect([2 2]), 'r', 'Linewidth', 2)
                plot([imrect(1) imrect(1) + imrect(3)], imrect(2)+imrect([4 4]), 'r', 'Linewidth', 2)
                plot(imrect([1 1]),[imrect(2), imrect(2) + imrect(4)], 'r', 'Linewidth', 2)
                plot(imrect(1)+imrect([3 3]), [imrect(2), imrect(2) + imrect(4)], 'r', 'Linewidth', 2)
                
                subplot(132, 'align')
            end
            
            if ~all(size(I)==dims(1))
                continue
            end
            
            Iwin = (I - mean(I(:))).*hwin;
            
            if ip.Results.makeMovie
                imagesc(Iwin, [-1 1]*max(abs(Iwin(:))))
                axis off
            end
            
            
            fIm = fftshift(fft2(Iwin));
            xax = -dims(1)/2:dims/2;
            xax = xax / dims(1) * ppd;
            
            if ip.Results.makeMovie
                subplot(133, 'align')
                imagesc(xax, xax, abs(fIm))
                xlabel('SF_x')
                ylabel('SF_y')
                drawnow
                
                currFrame = getframe(gcf);
                writeVideo(vidObj, currFrame)
            end
            
            fixIms(:,:,thisfix,2) = Iwin;
            fftIms(:,:,thisfix,2) = fIm;
            
        end
        
        
    end
    
    close(vidObj)
    
    
    %% Loop over units in group
    
    clist = find(clusts==clustGroup);
    
    for cc = clist(:)'
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
        if ip.Results.plot
            figure(1); clf
            set(gcf, 'Color', 'w')
            subplot(2,2,1)
            imagesc(xax, xax, rf)
            title('RF')
            xlabel('sf')
            ylabel('sf')
        end
        freshape = abs(reshape(fftIms(:,:,:,2), [prod(dims) nfix]));
        
        fproj = freshape'*rf(:);
        
        good_trials= find(sum(freshape)~=0);
        if ip.Results.plot
            subplot(2,2,2)
            
        end
        h = histogram(fproj(good_trials), 'FaceColor', .5*[1 1 1]);
        
        levels = prctile(fproj(good_trials), [10 90]);
        
        lowix = good_trials(fproj(good_trials) < levels(1));
        hiix = good_trials(fproj(good_trials) > levels(2));
        
        if ip.Results.plot
            hold on
            plot(levels(1)*[1 1], ylim, 'r', 'Linewidth', 2)
            plot(levels(2)*[1 1], ylim, 'b', 'Linewidth', 2)
            
            xlabel('Generator Signal')
            ylabel('Fixation Count')
            
            % psth
            subplot(2,2,3)
        end
        
        spksfilt = filter(ones(sm,1), 1, squeeze(spks(:,cc,:))')';
        
        psthhi = mean(spksfilt(hiix,:))/binsize/sm;
        psthlow = mean(spksfilt(lowix, :))/binsize/sm;
        
        pstvhi = std(spksfilt(hiix,:))/binsize/sm;
        pstvlow = std(spksfilt(lowix, :))/binsize/sm;
        
        if ip.Results.plot
            plot.errorbarFill(lags, psthhi, pstvhi/sqrt(numel(hiix)), 'b', 'FaceColor', 'b'); hold on
            plot.errorbarFill(lags, psthlow, pstvlow/sqrt(numel(lowix)), 'r', 'FaceColor', 'r');
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
        end
        
        
        fftrf(cc).lags = lags;
        fftrf(cc).rfLocation = rfCenter;
        fftrf(cc).rateHi = psthhi;
        fftrf(cc).rateLow = psthlow;
        fftrf(cc).stdHi = pstvhi;
        fftrf(cc).stdLow = pstvlow;
        fftrf(cc).nHi = numel(hiix);
        fftrf(cc).nLow = numel(lowix);
        fftrf(cc).rf.kx = xax;
        fftrf(cc).rf.ky = xax;
        fftrf(cc).rf.Ifit = reshape(Ifit, size(kx));
        fftrf(cc).xproj.bins = h.BinEdges(1:end-1)+h.BinWidth/2;
        fftrf(cc).xproj.cnt = h.Values;
        fftrf(cc).xproj.levels = levels;
        fftrf(cc).cid = grf(cc).cid;
        
        [i,j] = find(squeeze(spks(ind,cc,:)));
        
        if ip.Results.plot
            figure(2); clf
            subplot(1,2,1)
            plot.raster(j,i,10);
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
            
            drawnow
        end
    end
end
