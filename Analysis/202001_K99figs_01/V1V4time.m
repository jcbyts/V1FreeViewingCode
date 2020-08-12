

%%
seed = 240; %randi(1e3)
rng(seed)

k = 50;
v1Kernel = [0 diff(normpdf(linspace(-3, 5, k)))];
% v1Kernel = v1Kernel.*exp( (1:k)/ -20);

v4Kernel = normpdf(linspace(-2.5, 5, k*2));
v4Kernel = v4Kernel.*exp( (1:k*2)/-20);
v1Kernel = v1Kernel ./ norm(v1Kernel);
v4Kernel = v4Kernel ./ norm(v4Kernel);
figure(2); clf
plot(v1Kernel); hold on
plot(v4Kernel)



figure(1); clf
im = imread('office_1.jpg');
im = histeq(im);
im = mean(im,3);
im = imgaborfilt(im, 2, 90);

imagesc(im); 


nFrames = 200;
dim = size(im);
win = [10 10];
ctr = [196 312];
eyepos = cumsum(.08*randn(nFrames,2)) + ctr;
saccadeTimes = [100 120];
for i = 1:numel(saccadeTimes)
eyepos(saccadeTimes(i):end,1) = eyepos(saccadeTimes(i):end,1) - 10;
end




M = zeros(win(1), win(1), nFrames);

for i = 1:nFrames
    rect = [eyepos(i,:) win-1];
    M(:,:,i) = imcrop(im, rect);
end

Mfilt = M;
for i = 1:win(1)
    for j = 1:win(2)
        Mfilt(i,j,:) = filter(v1Kernel, 1, squeeze(M(i,j,:)));
    end
end

Mfilt(:,:,1:k) = 0;

figure(1); clf
subplot(3,1,1)
plot(eyepos(:,1)); hold on

g = squeeze(Mfilt(1,2,:));
g = [g squeeze(Mfilt(10,2,:))];
g = g/10;
v1 = g.^2;

subplot(3,1,2)
plot(v1)

subplot(3,1,3)
v4 = filter(v4Kernel, 1, sum(v1,2)).^2;
v4sep = filter(v4Kernel, 1, v1).^2;

plot(v4sep); hold on

plot(v4)

saveas(1, fullfile('Figures', 'K99', 'timeExample.pdf'))
saveas(2, fullfile('Figures', 'K99', 'temporalRF.pdf'))
% plot((squeeze(mean(mean(Mfilt,2)))').^2)

