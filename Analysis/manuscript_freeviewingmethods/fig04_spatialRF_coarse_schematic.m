%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/
figDir = 'Figures/manuscript_freeviewing/fig04';

%% Panel A-C: How gaze-contingent retinotopic mapping works

% pick a session, where we can reconstruct full stimulus examples
Exp = io.dataFactoryGratingSubspace(56);

% index into Dot trials
validTrials = io.getValidTrials(Exp, 'Dots');

iTrial = validTrials(1);

frames = [500 700:10:800];

% full screen image
[Stim, frameInfo] = regenerateStimulus(Exp, iTrial, Exp.S.screenRect - [Exp.S.centerPix Exp.S.centerPix], 'spatialBinSize', 1, ...
    'includeProbe', true, 'frameIndex', frames, 'GazeContingent', false);

% gaze-contingent ROI
ppd = Exp.S.pixPerDeg;
ROI = [-14 -10 14 10];
[StimGC, ~] = regenerateStimulus(Exp, iTrial, round(ROI*ppd), 'spatialBinSize', 1, ...
    'includeProbe', true, 'frameIndex', frames, 'GazeContingent', true);

% most sessions showed only white dots, which is what's primarily reported in the
% text. Some sessions showed both black and white dots. Regardless, when we
% do retinotopic mapping, we ignore sign (assuming some level of complex
% cell
Stim = abs(Stim);
StimGC = abs(StimGC);

%% output

for i = 1:size(Stim,3)
    figure(1); clf
    imagesc(Stim(:,:,i), [-127 127]); colormap gray
    axis equal
    hold on
    plot(frameInfo.eyeAtFrame(i,2), frameInfo.eyeAtFrame(i,3), 'or')
    axis off
    
    plot.fixfigure(gcf, 10, [5 4])
    saveas(gcf, fullfile(figDir, sprintf('fig04_exampleframe%d.pdf', i)))
    
    drawnow
end
hold on
tstart = Exp.ptb2Ephys(Exp.D{iTrial}.STARTCLOCKTIME);
tstop = Exp.ptb2Ephys(Exp.D{iTrial}.ENDCLOCKTIME);

tt = Exp.vpx2ephys(Exp.vpx.smo(:,1));
ix = tt > tstart & tt < tstop;
ppd = Exp.S.pixPerDeg;
ctr = Exp.S.centerPix;
x = Exp.vpx.smo(ix,2)*ppd + ctr(1);
y = ctr(2) - Exp.vpx.smo(ix,3)*ppd;
x(Exp.vpx.Labels(ix) > 2) = nan;
plot(x, y, '-c', 'MarkerSize', 2)
plot.fixfigure(gcf, 10, [5 4])
saveas(gcf, fullfile(figDir, 'fig04_example_eyepos.pdf'))


%% example coarse grid
xax = 0:ppd:size(StimGC,2);
n = numel(xax);
dim = size(StimGC);
dim(3) = [];
SE = {};
xax = [round(xax+1), dim(1)];

for i = 1:size(StimGC,3)
    figure(1); clf
    imagesc(StimGC(:,:,i)); hold on
    for j = 1:n
        for k = 1:n
            plot(xax([j j]), ylim, 'r')
            plot(xlim, xax([k k]), 'r')
        end
    end
    
    xd = xlim;
    xlim(xd + [-10 10]);
    yd = ylim;
    ylim(yd + [-10 10]);
    axis off
    
    plot.fixfigure(gcf, 10, [5 4])
    saveas(gcf, fullfile(figDir, sprintf('fig04_ROI_coarse_grid_%d.pdf', i)))
    
    % average in coarse grid
    [xx,yy] = meshgrid(xax);
    I = zeros(size(xx));
    Iraw = StimGC(:,:,i);
    for ii = 1:n
        for jj = 1:n
            rows = xax(ii):xax(ii+1);
            cols = xax(jj):xax(jj+1);
            m = min(numel(rows), numel(cols));
            ind = sub2ind(dim, cols(1:m), rows(1:m));
            I(jj,ii) = mean(Iraw(ind));
        end
    end
    
    SE{i} = I;
    
    figure(2); clf
    imagesc(I, [-127 127]/4)
    colormap gray
    axis off
    plot.fixfigure(gcf, 10, [5 4])
    saveas(gcf, fullfile(figDir, sprintf('fig04_ROI_coarse_binned_%d.png', i)))
    
end

%% Plot the stimulus ensemble
nframes = numel(frames);
sz = size(SE{1});
I = reshape(cell2mat(SE), [sz nframes]);
for i = 1:nframes
    I(1,:,i) = -200;
    I(sz(1)-1,:,i) = -200;
    I(:,1,i) = -200;
    I(:,sz(1)-1,i) = -200;
    
end
I = permute(I, [3 2 1]);


xax = 1:sz(2);
yax = 1:sz(1);

cmap = gray;
[xx,tt,yy] = meshgrid(xax, (1:nframes)*8, yax);

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
% plot3(10+[10 16], [1 1]*39, -[50 50], 'r', 'Linewidth', 2)
% plot3(10+[10 16], [1 1]*30.5, -[50 50], 'r', 'Linewidth', 2)
% text(40, 32, -50, '0.1 d.v.a', 'Color', 'r')

plot.fixfigure(gcf, 8, [6 3])
saveas(gcf, fullfile(figDir, 'fig04_binned_course_ensemble.png'))


%%