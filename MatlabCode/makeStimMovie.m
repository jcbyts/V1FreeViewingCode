function makeStimMovie(Stim, frameInfo, filename, showEye, ROI)
% makeStimMovie(Stim, frameInfo, filename, showEye, ROI)

if nargin < 5
    ROI = [];
end

if nargin < 4
    showEye = true;
end

if isempty(ROI)
    showROI = false;
else
    showROI = true;
end


eyeAtFrame = frameInfo.eyeAtFrame(:,2:3);
% eyeAtFrame(:,2) = -eyeAtFrame(:,2);

iFrame = 6;
xax = frameInfo.rect(1):frameInfo.rect(3);
yax = frameInfo.rect(2):frameInfo.rect(4);

h = figure(1); clf;
f = imagesc(xax, yax, Stim(:,:,iFrame), [-127 127]); colormap gray
xlim(xax([1 end]))
ylim(yax([1 end]))
hold on
ix = -5:0;

if showEye
    eh = plot(eyeAtFrame(iFrame+ix,1), eyeAtFrame(iFrame+ix,2), 'c-', 'Linewidth', 2);
end

if showROI
    x = eyeAtFrame(iFrame+ix(end),1) + ROI(1);
    y = eyeAtFrame(iFrame+ix(end),2) + ROI(2);
    w = ROI(3) - ROI(1);
    h = ROI(4) - ROI(2);
    pos = [x y w h];

    hr = rectangle('Position', pos, 'EdgeColor', 'r');
end

axis tight manual % this ensures that getframe() returns a consistent size
% axis xy

vobj = VideoWriter(filename, 'MPEG-4');
vobj.FrameRate = 60;
vobj.Quality = 100;
open(vobj);

nFrames = size(Stim,3);

for iFrame = 7:nFrames
    f.CData = Stim(:,:,iFrame);
    if showEye
        eh.XData = eyeAtFrame(iFrame+ix,1);
        eh.YData = eyeAtFrame(iFrame+ix,2);
    end
    
    if showROI
        x = eyeAtFrame(iFrame+ix(end),1) + ROI(1);
        y = eyeAtFrame(iFrame+ix(end),2) + ROI(2);
        w = ROI(3) - ROI(1);
        h = ROI(4) - ROI(2);
        pos = [x y w h];

        hr.Position = pos;
    end
    xlim(xax([1 end]))
    ylim(yax([1 end]))
    drawnow

    frame = getframe(gca);
    writeVideo(vobj,frame);
end

close(vobj)
