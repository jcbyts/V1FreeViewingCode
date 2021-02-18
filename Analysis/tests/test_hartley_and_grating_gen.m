 
%%
pixperdeg = 30;
cpd = 3;
phase = 0;
ori = pi/2;

rPix = 50/2;

dPix = rPix*2 + 1;
[X,Y] = meshgrid(-rPix:rPix);
% Standard deviation of gaussian (e1)
sigma = dPix/8;

maxRadians = 2*pi * cpd /pixperdeg;

% Create the sinusoid (s1)
pha = phase * pi/180;
s1 = cos( cos(ori) * (maxRadians*Y) + ...
    sin(ori) * (maxRadians*X) + pha);

pixdims = (-dPix/2:dPix/2);
degdims = pixdims/pixperdeg;

figure(1); clf
imagesc(degdims, degdims, s1)



%% hartley
pixperdeg =  37.5048;
nOctaves = 6;
Freq0 = .5;
M   = 1;
freqs = sort([-2.^(0:(nOctaves-1))*Freq0 0 2.^(0:(nOctaves-1))*Freq0]);
kxs = freqs;
kys = freqs;

[kxgrid, kygrid]=meshgrid(kxs, kys);
        
[xx,yy] = meshgrid(pixdims);
twoPi = 2*pi/pixperdeg;

ori = ori + 2;
sf = 10;
[ky, kx] = pol2cart(ori/180*pi, sf);

% kx = 5;
% ky = 5;
% [ori, sf] = cart2pol(ky, kx);
disp(ori)

gr = twoPi * ((kx*xx + ky*yy) / M);
h = sin(gr) + cos(gr);

hi = h-mean(h(:));

figure(1); clf
subplot(1,2,1) % SPATIAL DOMAIN
imagesc(degdims, degdims, hi); hold on
colormap gray
plot(degdims, (kx/ky)*-degdims+median(degdims), 'r', 'Linewidth', 2)
xlabel('horizontal space')
ylabel('vertical space')
title(ori)

fi = abs(fftshift(fft2((hi))));

% axes in the fourier domain
x = -floor(dPix/2):floor((dPix-1)/2);
x = x/-x(1)*pixperdeg/2;

subplot(1,2,2) % FOURIER DOMAIN
imagesc(x, x, fi)
hold on
plot(kx, ky, 'or', 'Linewidth', 2)
xlabel('horizontal frequencies')
ylabel('vertical frequencies')
axis xy



