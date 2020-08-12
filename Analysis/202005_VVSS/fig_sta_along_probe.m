
sessId = 12;
[Exp,S] = io.dataFactory(sessId);

%% get unit locations
cids = Exp.osp.cids;
NC = numel(cids);
nChan = numel(Exp.osp.xc);

figure(1); clf
x = nan(NC,1);
y = nan(NC,1);
for cc = 1:NC
temp = Exp.osp.temps(:,:,cc);
temp = sum(temp.^4,2);
temp = temp/sum(temp);
x(cc) = Exp.osp.xcoords'*temp;
y(cc) = Exp.osp.ycoords'*temp;
end

plot(x, y, 'o')

%% get orientation tuning
[stim, Robs, opts] = io.preprocess_grating_subspace_data(Exp);
NT = size(stim{1},1);
NC = size(Robs,2);

% Xstim = NIM.create_time_embedding(stim{1}, params(1));
num_lags = opts.num_lags_stim;
Xstim = makeStimRows(stim{1}, num_lags);

Xd = [Xstim ones(NT,1)];
xy = Xd'*Robs;

staSubspace = xy(1:end-1,:);
staSubspace = staSubspace./sum(Xstim)';

ix = sum(Robs) > 50;
staSubspace = staSubspace(:,ix);
x = x(ix);
y = y(ix);
NC = numel(x);
uq = Exp.osp.uQ(ix);



%% get STA pixels
tmp = load(fullfile('Data', 'Cstim3_raw_output.mat'));
tmp2 = load(fullfile('Data', 'modelSTAs.mat'));


%%
figure(1); clf
set(gcf, 'Color', 'w')
im = imread('Analysis/202001_K99figs_01/E32-50-S1-L6.jpg');
axes('Position', [.1 .1 .1 .9])
imagesc(im); axis xy
axis off
axes('Position', [.3 .1 .1 .9])
imagesc(im); axis xy
axis off
for cc = find(sum(Robs)>100)
    px = x(cc)./1000 + .25;
    py = y(cc)./1550 + .1;
    if mod(cc,2)==0 && (y(cc)-y(cc-1)) < 10
        px = px - 0.05;
    end
    
    
    a = reshape(staSubspace(:,cc), [opts.num_lags_stim, prod(opts.dim)]);
    [~, peakLag] = max(mean(a,2));
    
    a = (reshape(a(peakLag,:), opts.dim));
    
    [~, peakSF] = max(mean(a));
    
    oriTuning = a(:,peakSF)*120;
    dsi(cc) = abs(exp(1i*2*opts.oris/180*pi)'*(oriTuning/sum(oriTuning)));
    if dsi(cc) > .1
        clr = 'r';
    else
        clr = .5*[1 1 1];
        continue
    end
        axes('Position', [px py .05 .025])
        plot(opts.oris, oriTuning, 'Color', clr); hold on
        yd = ylim;
        ylim([0 yd(2)])
%         plot([90 90], ylim, 'k')
        set(gca, 'Color', 'none', 'Box', 'off')
        yt = get(gca, 'YTick');
        set(gca, 'YTick', yt([end]), 'TickDir', 'out')
        set(gca, 'XTickLabel', '')
        set(gca, 'YTickLabel', '')
    
end

plot.fixfigure(gcf, 10, [1, 2], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', 'oriTuningOnProbe.pdf'))

%%
%%
figure(1); clf
set(gcf, 'Color', 'w')
im = imread('Analysis/202001_K99figs/E32-50-S1-L6.jpg');
axes('Position', [.1 .05 .1 1.2])
imagesc(im); axis xy
axis off
axes('Position', [.25 .05 .1 1.2])
imagesc(im); axis xy
axis off


probe1 = find(x<30 & uq > 9);
np = numel(probe1);
for cc = 1:np
    px = .23;
    py = cc/np/1.15 + .05;
       
    a = reshape(staSubspace(:,probe1(cc)), [opts.num_lags_stim, prod(opts.dim)]);
    [~, peakLag] = max(mean(a,2));
    
    a = (reshape(a(peakLag,:), opts.dim));
    
    [~, peakSF] = max(mean(a));
    
    oriTuning = a(:,peakSF)*120;
    dsi(probe1(cc)) = abs(exp(1i*2*opts.oris/180*pi)'*(oriTuning/sum(oriTuning)));
    if dsi(probe1(cc)) > .1
        clr = 'r';
    else
        clr = .5*[1 1 1];
        
    end
        axes('Position', [px py .03 .6/np])
        plot(opts.oris, oriTuning, 'Color', clr); hold on
        yd = ylim;
        ylim([0 yd(2)])
        xlim([0 180])
%         axis tight
%         plot([90 90], ylim, 'k')
        set(gca, 'Color', 'none', 'Box', 'off')
        yt = get(gca, 'YTick');
        set(gca, 'YTick', yd([end]), 'TickDir', 'out')
        set(gca, 'XTickLabel', '')
        set(gca, 'XTick', 0:45:180)
        set(gca, 'TickLength', [0.05 0.05])
        set(gca, 'YTickLabel', '')
    
end

%
probe2 = find(x>150 & uq > 9);
np = numel(probe2);
for cc = 1:np
    px = .37;
    py = cc/np/1.15 + .05;
       
    a = reshape(staSubspace(:,probe2(cc)), [opts.num_lags_stim, prod(opts.dim)]);
    [~, peakLag] = max(mean(a,2));
    
    a = (reshape(a(peakLag,:), opts.dim));
    
    [~, peakSF] = max(mean(a));
    
    oriTuning = a(:,peakSF)*120;
    dsi(probe2(cc)) = abs(exp(1i*2*opts.oris/180*pi)'*(oriTuning/sum(oriTuning)));
    if dsi(probe2(cc)) > .1
        clr = 'r';
    else
        clr = .5*[1 1 1];
        
    end
        axes('Position', [px py .03 .6/np])
        plot(opts.oris, oriTuning, 'Color', clr); hold on
        yd = ylim;
        ylim([0 yd(2)])
        xlim([0 180])
%         axis tight
%         plot([90 90], ylim, 'k')
        set(gca, 'Color', 'none', 'Box', 'off')
        yt = get(gca, 'YTick');
        set(gca, 'XTick', 0:90:180)
        set(gca, 'YTick', yd([end]), 'TickDir', 'out')
        set(gca, 'TickLength', [0.05 0.05])
        set(gca, 'XTickLabel', '')
        set(gca, 'YTickLabel', '')
    
end



np = numel(probe1);

for cc = 1:np

    a = squeeze(tmp.stas(:,:,probe1(cc)));
    a = (a - median(a(:))) / std(a(:));
    [~, peakLag] = max(mean(a));
    
    px = .19;
    py = cc/np/1.15 + .05;
       
    I = reshape(a(:,peakLag), dim);
    axes('Position', [px py .035 .75/np])
    imagesc(I, [-4 6])
    axis off
    
end




np = numel(probe2);

for cc = 1:np

    a = squeeze(tmp.stas(:,:,probe2(cc)));
    a = (a - median(a(:))) / std(a(:));
    [~, peakLag] = max(mean(a));
    
    px = .33;
    py = cc/np/1.15 + .05;
       
    I = reshape(a(:,peakLag), dim);
    axes('Position', [px py .035 .75/np])
    imagesc(I, [-4 6])
    axis off
    
end

%%
plot.fixfigure(gcf, 10, [4, 4], 'offsetAxes', false)
% saveas(gcf, fullfile('Figures', 'K99', 'oriTuningOnProbe.pdf'))

%%
figure, plot(x, y, 'o'); hold on
plot(x(probe1), y(probe1), 'o')
plot(x(probe2), y(probe2), 'o')

%%
figure(1); clf
for cc = 1:NC
    px = x(cc)./1000 + .25;
    py = .9 - y(cc)./1550;
    if mod(cc,2)==0 && (y(cc)-y(cc-1)) < 10
        px = px - 0.05;
    end
    
    a = squeeze(tmp.stas(:,:,(cc)));
    a = (a - median(a(:))) / std(a(:));
    [~, peakLag] = max(mean(a));
    
    I = reshape(a(:,peakLag), dim);
    axes('Position', [px py .035 .035])
    imagesc(I)
    axis off
end

figure(2); clf
plot(x, y, 'o')

%%


figure(1); clf
dim = [40 40];

probe1 = find(x<50);
np = numel(probe1);

for cc = 55

    a = squeeze(tmp.stas(:,:,probe1(cc)));
    a = (a - median(a(:))) / std(a(:));
    [~, peakLag] = max(mean(a));
    
    px = .22;
    py = cc/np/1.15 + .05;
       
    I = reshape(a(:,peakLag), dim);
    axes('Position', [px py .035 .035])
    imagesc(I)
    axis off
    
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
% a = a / std(a(:));
% a = (a - min(a(:)))./(max(a(:)) - min(a(:)));
clim = max(abs(a(:)))*[-1 1]*.75;    
for ilag = 1:nlags
    subplot(1,nlags,ilag, 'align')
    imagesc(xax, yax, reshape(a(:,ilag), dims), clim)
    axis xy
%     contourf(reshape(a(:,ilag), [40 40]),[-4:.1:-2  0 2:6], 'Linestyle', 'none')
end
title(cc)
colormap(gray)

%%

figure(2); clf
I = reshape(mean(a(:,4:5),2), dims);
imagesc(xax, yax, I, clim)
axis xy
colormap gray
    

w = hanning(dims(1))*hanning(dims(2))';
I = w.*I;
I = I - mean(I(:));
% I = imresize(I, 10);
imagesc(fftshift(abs(fft2(I))))
colormap parula

%%

cmap = gray;
[xx,tt,yy] = meshgrid(xax, (1:nlags)*8, yax);
% [xx,yy] = meshgrid(xax, yax);
I = reshape(a, [dims nlags]);
I = permute(I, [3 2 1]);

figure(2); clf
set(gcf, 'Color', 'w')
h = slice(xx,tt, yy, I, [], (1:10)*8,[]);
set(gca, 'CLim', [-9 9])
for i = 1:numel(h)
    h(i).EdgeColor = 'none';
%     text(-12, i*8-4, 5, sprintf('%i ms', (i)*8))
end
view(79,11)
colormap(cmap)
axis off

hold on
plot3(10+[10 16], [1 1]*39, -[50 50], 'r', 'Linewidth', 2)
% plot3(10+[10 16], [1 1]*30.5, -[50 50], 'r', 'Linewidth', 2)
% text(40, 32, -50, '0.1 d.v.a', 'Color', 'r')

plot.fixfigure(gcf, 8, [14 3])
saveas(gcf, fullfile('Figures', 'K99', sprintf('sta%02.0f.png', cc)))
%%
% imagesc(I(:,:,5))
figure(2); clf
for i = 4:5
surf(xx, yy, squeeze(I(:,:,i)+i*100), 'Linestyle', 'none'); hold on
end
colormap parula


%%

figure(2); clf

np = numel(probe2);
polar(0, 1.0, 'o'); hold on

for cc = 1:np

    a = squeeze(tmp.stas(:,:,probe2(cc)));
    a = (a - median(a(:))) / std(a(:));
    [~, peakLag] = max(mean(a));
    
    px = .33;
    py = cc/np/1.15 + .05;
       
    I = reshape(a(:,peakLag), dim);
    
    [xc,yc] = radialcenter(I.^4);
    xc = interp1(1:dim(1), xax/60, xc);
    yc = interp1(1:dim(1), yax/60, yc);
    [th, rho] = cart2pol(xc, yc);
    polar(th, rho, '.r'); hold on
    
end


np = numel(probe1);

for cc = 1:np

    a = squeeze(tmp.stas(:,:,probe1(cc)));
    a = (a - median(a(:))) / std(a(:));
    [~, peakLag] = max(mean(a));
    
    px = .33;
    py = cc/np/1.15 + .05;
       
    I = reshape(mean(a(:,peakLag+(-1:0)),2), dim);
    I = imgaussfilt(I,1);
    [xc,yc] = radialcenter(I.^4);
    xc = interp1(1:dim(1), xax/60, xc);
    yc = interp1(1:dim(1), yax/60, yc);
    [th, rho] = cart2pol(xc, yc);
    h = polar(th, rho, '.b'); hold on
    
end

% xlim([0 1])
plot.fixfigure(gcf, 10, [3 3])
% saveas(gcf, fullfile('Figures', 'K99', 'RFlocations.pdf'))

%%

[xx,yy] = meshgrid(linspace(-1,1,1000));

th = pi/2;
fr = 75;
I = sin( fr*(cos(th)*xx + sin(th)*yy));
figure(1); clf 
imagesc(I, [-2 2])
axis off



%%

figure(1); clf
iix = 1329000 + (1:5e3);
plot(Exp.vpx.smo(iix,2), Exp.vpx.smo(iix,3), 'c')
xlim([-10 10])
ylim([-10 10])
plot.fixfigure(gcf, 10, [3 3])
saveas(gcf, fullfile('Figures', 'K99', 'Eyepos1.pdf'))