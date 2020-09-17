
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);


%% load data

close all
sessId = sessId + 1;
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'kilowf', 'cleanup_spikes', 0);

% eyePosOrig = Exp.vpx.smo(:,2:3);
% 
% eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);
% 
% lam = .5; % mixing between original eye pos and corrected eye pos
% Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);


% get visually driven units

spkS = io.get_visual_units(Exp, 'plotit', true);

% plot spatial RFs to try to select a ROI
unit_mask = 0;
NC = numel(spkS);
hasrf = find(~isnan(arrayfun(@(x) x.x0, spkS)));
figure(2); clf
set(gcf, 'Color', 'w')
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

xax = spkS(1).xax/Exp.S.pixPerDeg;
yax = spkS(1).yax/Exp.S.pixPerDeg;
cc = lines(NC);
for cc = 1:NC %hasrf(:)'
    subplot(sx,sy,cc,'align')
    rf = abs(spkS(cc).unit_mask)/max(abs(spkS(cc).unit_mask(:)));
    
    I = spkS(cc).srf;
    I = I - mean(I);
    I = I / max(abs(I(:)));
    if isnan(sum(I(:)))
        I = zeros(size(I));
    else
        unit_mask = unit_mask + rf;
    end
    imagesc(xax, yax, I, [-1 1]); hold on
    colormap parula
    axis xy
    plot([0 0], ylim, 'w')
    plot(xlim,[0 0], 'w')
%     xlim([0 40])
%     ylim([-40 0])
%     [~, h] = contour(xax, yax, rf, [.75:.05:1], 'Color', cmap(cc,:)); hold on
end

figure(1); clf
ppd = Exp.S.pixPerDeg;
imagesc(xax*ppd, yax*ppd,unit_mask); axis xy
[i,j] = find(unit_mask>NC);
[min(xax(j)) max(xax(j))]

%%

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', [-100 -100 100 100], 'binSize', 10, 'eyePos', eyePos);


%%
cc = cc + 1;
nlags = 15;
sta = simpleRevcorr(Xstim, RobsSpace(:,cc)-mean(RobsSpace(:,cc)), nlags);
thresh = sqrt(robustcov(sta(:)))*4;
        
[sm,la] = bounds(sta(:));
        
        
figure(1); clf
subplot(2,4,4, 'align')
imagesc(sta)
            
figure(3); clf
            
plot(sta, 'Color', .5*[1 1 1 1])
hold on
plot(xlim, thresh*[1 1], 'r--')
plot(xlim, -thresh*[1 1], 'r--')

        
sta_ = nan(size(sta));
        
sta_(abs(sta)>thresh) = sta(abs(sta)>thresh);
        
plot(sta_, 'r')%sta > thresh
        
        
if abs(sm) > la
    sta_ = sta_ ./ sm;
else
    sta_ = sta ./ la;
end
        
sta_(isnan(sta_)) = 0;
sta_ = max(sta_, 0);
        
        
% find spatial RF
[~, bestlag] = max(std(sta_,[],2));

I = imgaussfilt(reshape(sta_(bestlag,:), opts.dims),.5);
srf = reshape(sta(bestlag,:), opts.dims);
        
[x0,y0] = radialcenter(I);
        
unit_mask = exp(-hypot((1:opts.dims(1)) - x0, (1:opts.dims(2))' - y0)) .* srf;
        
        
figure(5); clf
imagesc(opts.xax, opts.yax, srf)


