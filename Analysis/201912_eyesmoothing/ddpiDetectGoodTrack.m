
Q = ddpiReadFile('Z:\Data\ForageMapping\Logan20191219\ForageProceduralNoise_Logan_191219_01.ddpi');

% extract relevant data
p1x = Q(3,:)';
p1y = Q(4,:)';

p4x = Q(7,:)';
p4y = Q(8,:)';


% h = fvtool(b,a); 



figure(1); clf
subplot(1,2,1)
plot(p1x, p4x, '.');
axis tight
hold on
xlabel('P1')
ylabel('P4')
title('Horizontal Position')

mdlrx = fitlm(p1x,p4x,'RobustOpts','on');
mdlry = fitlm(p1y,p4y,'RobustOpts','on');
xcoef = mdlrx.Coefficients.Variables;
ycoef = mdlry.Coefficients.Variables;

xd = xlim;
plot(xd, xd*xcoef(2,1) + xcoef(1,1), 'r')

subplot(1,2,2)
plot(p1y, p4y, '.');
axis tight
hold on
xlabel('P1')
ylabel('P4')
title('Vertical Position')
xd = xlim;
plot(xd, xd*ycoef(2,1) + ycoef(1,1), 'r')

% use residuals to find outliers
p4xhat = p1x*xcoef(2,1) + xcoef(1,1);
residx = p4x - p4xhat;

p4yhat = p1y*ycoef(2,1) + ycoef(1,1);
residy = p4y - p4yhat;

outliers = false(n,1);
for i = 1:5
    z = hypot(zscore(residx(~outliers)),zscore(residy(~outliers)));
    outliers(~outliers) = z > 4;
    sum(outliers)
end

figure(2); clf
subplot(1,2,1)
plot(p1x, p4x, '.')
hold on
plot(p1x(~outliers), p4x(~outliers), '.')
xlabel('P1')
ylabel('P4')
title('X position')

subplot(1,2,2)
plot(p1y, p4y, '.')
hold on
plot(p1y(~outliers), p4y(~outliers), '.')
xlabel('P1')
ylabel('P4')
title('Y position')
legend({'outliers', 'good'})

figure(3); clf
gazex = p4x - p1x;
gazey = p4y - p1y;

plot(gazex); hold on
plot(find(~outliers), gazex(~outliers), '.')
xlabel('Sample #')
ylabel('Position')

[~, ind] = sort(p1x);

figure(4); clf
plot(p1x(~outliers), p4x(~outliers) - p4xhat(~outliers), '.'); hold on
plot(xlim, [0 0], 'k--')

%%

Fs = 1e3 ./ median(diff(Q(2,:)));
order = 3;


[b,a] = butter(order, 100/Fs, 'low');

x = p4x(~outliers);
x(isnan(x)) = [];

figure(1); clf,
plot(x); hold on
plot(filtfilt(b,a,x)); hold on
plot(sgolayfilt(x, 2, 15))


%%

bins = {1:720, 1:720};
C = hist3([p1x(:), p4x(:)], bins);
imagesc(bins{1}, bins{2}, (C)'); colormap parula