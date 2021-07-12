function plotGratingData(D)
% simple utility function for plotting grating data

figure(111); clf
subplot(3,1,1)
plot.raster(D.spikeTimes, D.spikeIds, 1); hold on
t = D.GratingOnsets;
plot((t*[1 1])', (ones(numel(t),1)*ylim)', 'r', 'Linewidth', 2); hold on
t = D.GratingOffsets;
plot((t*[1 1])', (ones(numel(t),1)*ylim)', 'g', 'Linewidth', 2)
title('Spikes + Gratings')

subplot(3,1,2)
plot(D.eyeTime, D.eyePos(:,1), 'k')
hold on
plot(D.eyeTime, D.eyePos(:,2), 'Color', .5*[1 1 1])
plot(D.eyeTime(D.eyeLabels==2), D.eyePos(D.eyeLabels==2, 1), 'r.')
title('Eye Position')

subplot(3,1,3)
plot(D.treadTime, D.treadSpeed, 'k', 'Linewidth', 2); hold on
title('Treadmill')
xlabel('Time (seconds)')