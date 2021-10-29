function fftrf = fixrsvp_by_fftrf(Exp, srf, grf, varargin)
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

stimulusSet = 'FixRsvpStim';
validTrials = io.getValidTrials(Exp, stimulusSet);

numFramesTrial = cellfun(@(x) size(x.PR.NoiseHistory,1), Exp.D(validTrials));
[~, longestTrial] = max(numFramesTrial);
goodTrials = longestTrial;

validTrials = validTrials(goodTrials);
numFramesTrial = numFramesTrial(goodTrials);
nFramesTotal = sum(numFramesTrial);


% get sample image to get dimensions right
hObj = stimuli.gaussimages(0, 'bkgd', Exp.S.bgColour, 'gray', false);
hObj.loadimages('rsvpFixStim.mat');
hObj.position = [0,0]*Exp.S.pixPerDeg + Exp.S.centerPix;
hObj.radius = round(Exp.D{validTrials(1)}.P.faceRadius*Exp.S.pixPerDeg);
hObj.imagenum = 1;
im = hObj.getImage();
dims = size(im);

fixIms = zeros(nFramesTotal, prod(dims));
fftIms = zeros(nFramesTotal, prod(dims));

frameIter = 0;

for thisTrial = validTrials(:)'
    
    hObj = stimuli.gaussimages(0, 'bkgd', Exp.S.bgColour, 'gray', false);
    hObj.loadimages('rsvpFixStim.mat');
    hObj.position = [0,0]*Exp.S.pixPerDeg + Exp.S.centerPix;
    hObj.radius = round(Exp.D{thisTrial}.P.faceRadius*Exp.S.pixPerDeg);
    
    noiseFrames = Exp.ptb2Ephys(Exp.D{thisTrial}.PR.NoiseHistory(:,1));
    nFrames = numel(noiseFrames);
    
    for iFrame = 1:nFrames
        thisFrame = iFrame + frameIter;
        
        % update image
        hObj.imagenum = Exp.D{thisTrial}.PR.NoiseHistory(iFrame,4);
        hObj.position = Exp.D{thisTrial}.PR.NoiseHistory(iFrame,2:3);
        
        im = hObj.getImage();
        fim = abs(fftshift(fft2(im-mean(im(:)))));
        % if plot
        %     figure(1); clf
        % subplot(121)
        % imagesc(im)
        % get fft power
        
        % subplot(122)
        % imagesc(fim)
        %     end
        %
        
        % store
        fixIms(thisFrame,:) = im(:);
        fftIms(thisFrame,:) = fim(:);
    end
    
    frameIter = frameIter + nFrames;
end

%% get unit RFs
ppd = Exp.S.pixPerDeg;
dims = size(fim);
xax = -dims(1)/2:dims/2;
xax = xax / dims(1) * ppd;

NC = numel(grf);
Ifit = zeros(prod(dims), NC);
for cc = 1:NC
        [kx, ky] = meshgrid(xax(2:end), xax(2:end));
        [ori, cpd] = cart2pol(kx, ky);
        ori = wrapToPi(ori);
        
        params = [grf(cc).rffit.oriBandwidth/180*pi, ...
            grf(cc).rffit.oriPref/180*pi, ...
            grf(cc).rffit.sfPref,...
            grf(cc).rffit.sfBandwidth,...
            grf(cc).rffit.amp, ...
            grf(cc).rffit.base];
        
        Ifit(:,cc) = prf.parametric_rf(params, [ori(:), cpd(:)]);
end


%%

xproj = fftIms*Ifit;
figure(1); clf
plot(xproj-mean(xproj))

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

    
    fixIms = zeros(dims(1), dims(2), nfix, 2);
    fftIms = zeros(dims(1), dims(2), nfix, 2);
    
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
