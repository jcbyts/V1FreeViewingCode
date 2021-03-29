%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/
figDir = 'Figures/manuscript_freeviewing/fig04';

%% Panel A-C: How gaze-contingent retinotopic mapping works
% Run this if you need to regenerate figures

% pick a session, where we can reconstruct full stimulus examples
Exp = io.dataFactory(56);

% index into Dot trials
validTrials = io.getValidTrials(Exp, 'Dots');

iTrial = validTrials(1);

frames = [700:20:900];

% full screen image
[Stim, frameInfo] = regenerateStimulus(Exp, iTrial, Exp.S.screenRect - [Exp.S.centerPix Exp.S.centerPix], 'spatialBinSize', 1, ...
    'includeProbe', true, 'frameIndex', frames, 'GazeContingent', false);

% gaze-contingent ROI
ppd = Exp.S.pixPerDeg;
ROI = [-14 -10 14 10];
% [StimGC, ~] = regenerateStimulus(Exp, iTrial, round(ROI*ppd), 'spatialBinSize', 1, ...
%     'includeProbe', true, 'frameIndex', frames, 'GazeContingent', true);

% most sessions showed only white dots, which is what's primarily reported in the
% text. Some sessions showed both black and white dots. Regardless, when we
% do retinotopic mapping, we ignore sign (assuming some level of complex
% cell
Stim = abs(Stim);
% StimGC = abs(StimGC);

% %% output
% 
% for i = 1:size(Stim,3)
%     figure(1); clf
%     imagesc(Stim(:,:,i), [-127 127]); colormap gray
%     axis equal
%     hold on
%     plot(frameInfo.eyeAtFrame(i,2), frameInfo.eyeAtFrame(i,3), 'or')
%     axis off
%     
%     plot.fixfigure(gcf, 10, [5 4])
%     saveas(gcf, fullfile(figDir, sprintf('fig04_exampleframe%d.pdf', i)))
%     
%     drawnow
% end
% hold on
% tstart = Exp.ptb2Ephys(Exp.D{iTrial}.STARTCLOCKTIME);
% tstop = Exp.ptb2Ephys(Exp.D{iTrial}.ENDCLOCKTIME);
% 
% tt = Exp.vpx2ephys(Exp.vpx.smo(:,1));
% ix = tt > tstart & tt < tstop;
% ppd = Exp.S.pixPerDeg;
% ctr = Exp.S.centerPix;
% x = Exp.vpx.smo(ix,2)*ppd + ctr(1);
% y = ctr(2) - Exp.vpx.smo(ix,3)*ppd;
% x(Exp.vpx.Labels(ix) > 2) = nan;
% plot(x, y, '-c', 'MarkerSize', 2)
% plot.fixfigure(gcf, 10, [5 4])
% saveas(gcf, fullfile(figDir, 'fig04_example_eyepos.pdf'))

ds = 3; % downsample to make things easier to plot
ppd = ppd/ds;

Stim = Stim(1:ds:end,1:ds:end,:);

%%

dim = size(Stim);
nframes = dim(3);
dim(3) = [];

binSize = 0.3;
xax = ROI(1):binSize:ROI(3);
yax = ROI(2):binSize:ROI(4);
xax = round(xax*ppd);
yax = round(yax*ppd);

fig = figure(1); clf
set(gcf, 'Color', 'w')
fig.Position = [100 100 650 300];


% -- STIM WITH ROI
ax = axes('Position', [.025 .2 .3 .75]);


xd = 1:dim(2);
yd = 1:dim(1);

dt = 20;
[xx,yy] = meshgrid(xd, yd);

alphas = linspace(1, .25, nframes);

h = [];
f = [];
xoffsets = linspace(0, 200, nframes);
yoffsets = linspace(0, 200, nframes);
for iframe = 1:nframes
    h = surf(xx+xoffsets(iframe), zeros(size(xx))+iframe*dt, yy+yoffsets(iframe), Stim(:,:,iframe)); hold on
    h.FaceAlpha = alphas(iframe);
%     fill3(xd([1 1 end end])+xoffsets(iframe), iframe*dt*[1 1 1 1], yd([1 end end 1])+yoffsets(iframe), 'k'); hold on %, 'FaceColor', 'None', 'EdgeColor', 'k')
end

axis equal
view(0, -5)

colormap gray
shading flat
caxis([-1 1])

% eye trace
plot3(frameInfo.eyeAtFrame(:,2)/ds + xoffsets(:), (1:nframes)*dt-5, frameInfo.eyeAtFrame(:,3)/ds + yoffsets(:), 'c', 'Linewidth', 2)

% ROI and bounding box
f = [];
for i = 1:nframes
    ex = frameInfo.eyeAtFrame(i,2)/ds;
    ey = frameInfo.eyeAtFrame(i,3)/ds;
    f(i) = fill3(xax([1 1 end end])+ex+xoffsets(i), dt*i*[1 1 1 1]-1, yax([1 end end 1])+ey+yoffsets(i), 'r', 'FaceColor', 'r', 'FaceAlpha', alphas(i)/3, 'EdgeColor', 'r');
     
    fill3(xd([1 1 end end])+xoffsets(i), i*dt*[1 1 1 1], yd([1 end end 1])+yoffsets(i), 'k', 'FaceColor', 'None', 'EdgeColor', 'k')
end

set(f(1), 'Linewidth', 2)

xlabel('Azimuth')
ylabel('Time')
zlabel('Elevation')

axis equal
axis off

% --- Annotations
% slope = ax.Position(4)./ax.Position(3);
% annotation('textbox', [.15 .8 .1 .1], 'string', 'Gaze-Contingent ROI', 'EdgeColor', 'none', 'Fontsize', 8, 'Color', 'r');
% annotation('textbox', [.05 .32 .1 .1], 'string', 'Eye Position', 'EdgeColor', 'none', 'Fontsize', 8, 'Color', 'c');
% x0 = [.26 .34];
% y0 = .36;
% y0 = [y0 y0 + .8*diff(x0)*slope];
% theta = cart2pol(diff(x0)*fig.Position(3), diff(y0)*fig.Position(4))/pi*180;
% a = annotation('textarrow', x0, y0, 'String', 'Time', 'TextRotation', theta, 'FontSize', 8, 'HeadStyle', 'vback1', 'HeadWidth', 5, 'HeadLength', 5, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'cap');

% annotation('arrow', [.35 .42], [.6 .6], 'HeadStyle', 'vback1', 'HeadWidth', 5, 'HeadLength', 5)
% annotation('textbox', [.33 .6 .1 .1], 'string', 'Re-sample', 'EdgeColor', 'none', 'Fontsize', 8, 'Color', 'k');
% annotation('textbox', [.01 .85 .1 .1], 'string', 'A', 'Fontsize', 14, 'EdgeColor', 'none')

% -- Resampled stimulus
ax = axes('Position', [.47 .2 .28 .75]);

n = numel(xax);
m = numel(yax);
stimGC = zeros(m,n,nframes);

for iframe = 1:nframes
    Ifull = Stim(:,:,iframe);
    for j = 1:n-1
        for k = 1:m-1
            jj = find( (xax(j) + ex) < xd & (xax(j+1) + ex) > xd);
            kk = find( (yax(k) + ey) < yd & (yax(k+1) + ey) > yd);
            
            if isempty(jj) || isempty(kk)
                continue
            end
            
            [kk, jj] = meshgrid(kk, jj);
            ind = sub2ind(dim, kk(:), jj(:));
            stimGC(k,j,iframe) = mean(Ifull(ind));
            
        end
    end
end

% imagesc(stimGC(:,:,1), [-1 1])
xoffsets = linspace(0, 50, nframes);
yoffsets = linspace(0, 30, nframes);
[xx,yy] = meshgrid(1:n,1:m);
for iframe = 1:nframes
    h = surf(xx+xoffsets(iframe), zeros(size(xx))+iframe*dt, yy+yoffsets(iframe), round(stimGC(:,:,iframe))); hold on
    h.FaceAlpha = alphas(iframe);
    
end


axis equal
view(0, -5)

colormap gray
shading flat
caxis([-1 1])

for i = 1:nframes
    
    fill3([1 1 n n]+xoffsets(i), i*dt*[1 1 1 1], [1 m m 1]+yoffsets(i), 'k', 'FaceColor', 'None', 'EdgeColor', 'r'); hold on
    drawnow
end

axis off
set(gcf, 'PaperSize', [4.5 2], 'PaperPosition', [0 0 4.5 2])


%%
fname = fullfile(figDir, 'fig04_schematic.png');
saveas(gcf, fname);