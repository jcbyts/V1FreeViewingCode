
sessId = 12;
[Exp, S] = io.dataFactory(sessId);


%% get valid trials

stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);

%% build V1 filter
k = 50;

% gaussian derivastive
v1Kernel = [0 diff(normpdf(linspace(-4, 12, k)))];

v1Kernel = v1Kernel ./ norm(v1Kernel);

figure(2); clf
plot(v1Kernel); hold on

sfs = sort([1 10]);
nSfs = numel(sfs);

ori = 45;


winsize = 3;
win = winsize*ceil(Exp.S.pixPerDeg)*[1 1];
xax = linspace(-winsize/2, winsize/2, win(1));
[xx, yy] = meshgrid(xax);

xg = cosd(ori)*xx + sind(ori)*yy;
mag = hanning(win(1))*hanning(win(1))';

wtsCos = zeros(prod(win), nSfs);
wtsSin = zeros(prod(win), nSfs);
figure(1); clf
for iSf = 1:nSfs
    
    tmp = reshape(cos(sfs(iSf)*xg*winsize*2).*mag, [], 1);
    wtsCos(:,iSf) = tmp/norm(tmp);
    
    tmp = reshape(sin(sfs(iSf)*xg*winsize*2).*mag, [], 1);
    wtsSin(:,iSf) = tmp/norm(tmp);
    
    subplot(2,nSfs,iSf)
    imagesc(xax, xax, reshape(wtsCos(:,iSf), win))
    title(sfs(iSf))
    subplot(2,nSfs,nSfs+iSf)
    imagesc(xax, xax, reshape(wtsSin(:,iSf), win))
    
end
%% run analysis

nTrials = 4;
trial = repmat(struct('nFrames', [], 'sacTimes', [], 'sacSizes', [], 'Mcrop', []), nTrials, 1);

fprintf('Cropping gaze contingent window now\n')
for iTrial = 1:nTrials
    fprintf('%d/%d\n', iTrial, nTrials)
    
    thisTrial = validTrials(iTrial);

    % get eye position
    inds = find(Exp.vpx.smo(:,1) > Exp.D{thisTrial}.START_VPX &...
    Exp.vpx.smo(:,1) < Exp.D{thisTrial}.END_VPX);

    eyepos = [Exp.vpx.smo(inds,2) Exp.vpx.smo(inds,3)];
    eyepos = sgolayfilt(eyepos, 1, 9);

    % convert to pixels in image
    eyepos = eyepos * Exp.S.pixPerDeg + Exp.S.centerPix;

    % load image
    try
        Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));
    catch
        trial(iTrial).nFrames = 0;
        continue
    end
    Im = imresize(Im, Exp.S.screenRect([4 3]));
    Im = mean(Im,3); % grayscale
    
    % find saccades
    iix = Exp.slist(:,4) > inds(1) & Exp.slist(:,5) < inds(end);
    
    % convert to sample index into trial
    slist = Exp.slist(iix,:);
    slist(:,4:6) = slist(:,4:6) - inds(1) + 1;

    sacTimes = slist(:,4);
    
    dx = eyepos(slist(:,5),1)-eyepos(slist(:,4),1);
    dy = eyepos(slist(:,5),2)-eyepos(slist(:,4),2);
    sacSizes = hypot(dx, dy)/Exp.S.pixPerDeg;

    nFrames = size(eyepos,1);
    
    M = zeros(nFrames, prod(win));
    
    
    for i = 1:nFrames
        try
            rect = [eyepos(i,:) win-1];
            tmp = imcrop(Im, rect);
            M(i,:) = tmp(:);
        catch
            disp('error cropping')
        end
        
    end
    
    
    trial(iTrial).sacSizes = sacSizes;
    trial(iTrial).sacTimes = sacTimes;
    trial(iTrial).nFrames = nFrames;
    trial(iTrial).Mcrop = M;
end

%% concatenate data
n = (arrayfun(@(x) x.nFrames, trial));
trial(n == 0) = [];
nTrials = numel(trial);
n = [0; cumsum(n(n~=0))];

sacTimes = [];
sacSizes = [];
M = cell2mat(arrayfun(@(x) x.Mcrop, trial, 'uni', 0));
for iTrial = 1:nTrials
    sacTimes = [sacTimes; trial(iTrial).sacTimes + n(iTrial)];
    sacSizes = [sacSizes; trial(iTrial).sacSizes];
end

%% filter
figure(1); clf
subplot(1,2,1)
imagesc(M(1:1e3,:))
subplot(1,2,2)
Mfilt = filter(v1Kernel, 1, M);
plot(Mfilt(1:1e3,:))

%%

E = (M*wtsCos).^2 + (M*wtsSin).^2;
clf
plot(E)
%% plot
figure(1); clf
set(gcf, 'DefaultAxesColorOrder', lines)
% ss = prctile(sacSizes, 0:25:100);
ss = [0 1 5 8 10];

for i = 1:nSfs
    subplot(1,nSfs,i)
    for j = 1:numel(ss)-1
    
        ix = sacSizes > ss(j) & sacSizes < ss(j+1);
        x = E(:,i);
%         x = filter(v1Kernel,1,M(:,i));
        [m, ~, lags] = eventTriggeredAverage(x, sacTimes(ix), 100*[-1 1]);
        plot(lags, m); hold on
    end
    
end



%%





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

