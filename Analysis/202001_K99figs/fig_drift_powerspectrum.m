

Fs = 540;
xvel = [0; diff(imgaussfilt(xpos, 9))]*Fs;
yvel = [0; diff(imgaussfilt(ypos, 9))]*Fs;



figure(1); clf
speed = hypot(xvel, yvel);
plot(speed);

fixations = Exp.vpx.Labels==1 & speed < 3;

fixon = find(diff(fixations)==1);
fixoff = find(diff(fixations)==-1);

if fixations(1)
    fixon = [1; fixon];
end

if fixations(end)
    fixoff = [fixoff; numel(fixations)];
end

nFix = numel(fixon);
assert(nFix==numel(fixoff))


fixdur = fixoff - fixon;

rem = (fixdur/Fs) < .1;
fixon(rem) = [];
fixoff(rem) = [];
nFix = numel(fixon);

fprintf('Found %d fixations\n', nFix)

%%
xpos = Exp.vpx.smo(:,2);
ypos = Exp.vpx.smo(:,3);

xpos = sgolayfilt(Exp.vpx.smo(:,2), 1, 3);
ypos = sgolayfilt(Exp.vpx.smo(:,3), 1, 3);
tic
xd = nan(size(xpos));
yd = nan(size(ypos));
for iFix = 1:nFix
    iix = fixon(iFix):fixoff(iFix);
    xd(iix) = detrend(xpos(iix), 'constant');
    yd(iix) = detrend(ypos(iix), 'constant');
end
toc

%%
figure(1); clf
x = xd(~isnan(xd));
% x = imgaussfilt(x, 1);
y = yd(~isnan(yd));
% y = imgaussfilt(y, 1);
cmap = lines;
[Pxx, xax] = pwelch(x, [], [], [], Fs);
% plot(xax, 10*log10(imboxfilt(Pxx, 101)), 'Color', cmap(5,:)); hold on
plot(xax, 10*log10(imboxfilt(Pxx, 101)), 'Color', cmap(5,:)); hold on
[Pxx, xax] = pwelch(y, [], [], [], Fs);

% plot(xax, 10*log10(Pxx), 'Color', cmap(1,:))
plot(xax, 10*log10(imboxfilt(Pxx, 101)), 'Color', cmap(1,:)); hold on
xlim([.5 200])
set(gca, 'Xscale', 'log', 'box', 'off')
% figure, plot(xd)
xlabel('Frequency (Hz)')
ylabel('dB')

plot.fixfigure(gcf, 8, [4 4], 'offsetAxes', false)
saveas(gcf, fullfile('Figures', 'K99', 'drift_power.pdf'))

