
Im = imread('office_1.jpg');
Im = mean(Im,3); % grayscale
Im = imresize(Im, 2);
dim = size(Im);
figure(1); clf
imagesc(Im)

%%
seed = 240; %randi(1e3)
rng(seed)

k = 50;
v1Kernel = [0 diff(normpdf(linspace(-5, 7, k)))];

v1Kernel = v1Kernel ./ norm(v1Kernel);

figure(2); clf
plot(v1Kernel); hold on


%% build eye traces
clf

nFix = 400;
maxFixDur = 700;
nFrames = 300*maxFixDur;


win = [10 10]; % 100 neighboaring V1 cells

ctr = fliplr(dim/2); % start in the center of the screen
% eyepos = cumsum(0*randn(nFrames,2)) + ctr; % drift

ppd = 20;
sacrange = linspace(0, 10*ppd, 100);
pdfSac = normpdf(sacrange, 4*ppd, 3*ppd);
pdfSac = pdfSac / sum(pdfSac);
cdfSac = cumsum(pdfSac);
pdfFix = normpdf(1:maxFixDur, 100, 170).*(1-exp((1:maxFixDur)/-400));

pdfFix = pdfFix./sum(pdfFix);
cdfFix = cumsum(pdfFix);
figure(1); clf
plot(pdfFix)
%%
% plot(sacrange, pdfSac)

eyepos = ones(nFrames, 2).*ctr;
iFrame = 1;
% eyepos(iFrame,:) = ctr;

nextSaccade = ceil(interp1(cdfFix, 1:maxFixDur, rand())) + iFrame;
sacTimes = nextSaccade;
sacSizes = [];
for iFix = 1:nFix
%     
    % drift
    while iFrame < nextSaccade
        iFrame = iFrame + 1;
        eyepos(iFrame,:) = eyepos(iFrame-1,:) + 0*randn(1,2); % brownian motion
    end
    
    % saccade
%     sacSize = -sign(eyepos(iFrame,:)-ctr)*40;
    sacSize = interp1(cdfSac, sacrange, rand(1,2));
    if rand < .5
        sacSize = sacSize.*sign(randn(1,2));
    else
        sacSize = sacSize.*-sign(eyepos(iFrame,:)-ctr);
    end
    
    while any(isnan(sacSize))
        sacSize = interp1(cdfSac, sacrange, rand(1,2)).*-sign(eyepos(iFrame,:)-ctr);
    end
    eyepos(iFrame,:) = eyepos(iFrame,:) + sacSize;
    nextSaccade = ceil(interp1(cdfFix, 1:maxFixDur, rand())) + iFrame;
    sacTimes = [sacTimes nextSaccade];
    sacSizes = [sacSizes hypot(sacSize(1), sacSize(2))];
end

nFrames = iFrame;
eyepos = eyepos(1:iFrame,:);
eyepos(:,1) = imgaussfilt(eyepos(:,1), 3);
eyepos(:,2) = imgaussfilt(eyepos(:,2), 3);

figure(1); clf
plot(eyepos(:,1), eyepos(:,2), 'k')
xlim([0 dim(2)])
ylim([0 dim(1)])

figure(2); clf
plot(eyepos(:,1))
    


%% process image with V1 RFs of different sf tuning

figure(1); clf


sfs = [100 50];
ori = 0;


nSfs = numel(sfs);
MfiltR = cell(nSfs,1);
MfiltL = cell(nSfs,1);
figure(1); clf

for iSf = 1:nSfs
    
    sf = sfs(iSf);
    [mag, phase] = imgaborfilt(Im, sf, ori);
    im1 = mag.*cos(phase);
    im2 = mag.*sin(phase);
    dim = size(im1);
%     im = imgaussfilt(Im, sf);
    subplot(1,nSfs,iSf)
    imagesc(im1); hold on
    plot(eyepos(:,1), eyepos(:,2))
    drawnow
    ML = zeros(nFrames, prod(win));
    MR = zeros(nFrames, prod(win));
    
    for i = 1:nFrames
        rect = [eyepos(i,:) win-1];
        try
            tmp = imcrop(im1, rect);
            ML(i,:) = tmp(:);
            tmp = imcrop(im2, rect);
            MR(i,:) = tmp(:);
        end
    end
    
    
    MfiltL{iSf} = filter(v1Kernel, 1, ML);
    MfiltR{iSf} = filter(v1Kernel, 1, MR);

end

figure(2); clf
plot(MfiltL{1})
%%

figure(1); clf
% histogram(sacSizes)
levels = quantile(sacSizes, 0:.5:1);
ix = sacSizes > median(sacSizes);
set(gcf, 'DefaultAxesColorOrder', parula(numel(levels)));
for i = 1:nSfs
    M = MfiltL{i}.^2 + MfiltR{i}.^2;
    subplot(1,nSfs,i)
    for j = 1:numel(levels)-1
        ix = sacSizes > levels(j) & sacSizes < levels(j+1);
%         x = mean(max(MfiltL{i},0),2);
%         x = max(Mfilt{i}(:,2),0);
%         x = MfiltL{i}(:,1);
        x = M(:,1);
        m = eventTriggeredAverage(x, sacTimes(ix)', 50*[-1 1]);
        plot(m); hold on
    end
%     ylim([-1 2])
end

%%
