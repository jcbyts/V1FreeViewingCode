
%% quick and dirty simulation 


%% load experimental session
sessId = 12;
[Exp, S] = io.dataFactory(sessId);


%% get valid trials

stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);

%% temporal filters
k = 50;

Ntk = 100;
TMagno = MacaqueRetinaM_Temporal_BK();
TParvo = MacaqueRetinaP_Temporal_BK();

TMagno = TMagno(1:2:end);
TParvo = TParvo(1:2:end);

figure(1); clf
plot(TMagno); hold on
plot(TParvo);



%% spatial filters
ori = 0; %45;
sfs = [1 5 10];
winsize = 2; % degrees
win = ceil(winsize*[1 1]*Exp.S.pixPerDeg);
xax = linspace(-winsize/2, winsize/2, win(1));
[xx, yy] = meshgrid(xax);

xg = cosd(ori)*xx + sind(ori)*yy;
mag = hanning(win(1))*hanning(win(1))';
nSfs = numel(sfs);

wtsL = zeros(prod(win), nSfs);
wtsR = zeros(prod(win), nSfs);
figure(1); clf
for iSf = 1:nSfs
    
    tmp = reshape(cos(sfs(iSf)*xg*pi*winsize).*mag, [], 1);
    wtsL(:,iSf) = tmp/norm(tmp);
    
    tmp = reshape(sin(sfs(iSf)*xg*pi*winsize).*mag, [], 1);
    wtsR(:,iSf) = tmp/norm(tmp);
    
    subplot(2,nSfs,iSf)
    imagesc(xax, xax, reshape(wtsL(:,iSf), win))
    title(sfs(iSf))
    
    subplot(2,nSfs,nSfs+iSf)
    imagesc(xax, xax, reshape(wtsR(:,iSf), win))
end

%% run analysis
nTrials = 40;
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
    Im = mean(Im,3);
        
    figure(1); clf
    imagesc(Im); hold on
    plot(eyepos(:,1), eyepos(:,2), 'r')
    drawnow
    
    % find saccades
    iix = Exp.slist(:,4) > inds(1) & Exp.slist(:,5) < inds(end);
    
    % convert to sample index into trial
    slist = Exp.slist(iix,:);
    slist(:,4:6) = slist(:,4:6) - inds(1) + 1;

    sacTimes = slist(:,4:5);
    
    dx = eyepos(slist(:,5),1)-eyepos(slist(:,4),1);
    dy = eyepos(slist(:,5),2)-eyepos(slist(:,4),2);
    sacSizes = hypot(dx, dy)/Exp.S.pixPerDeg;

    nFrames = size(eyepos,1);
    
    M = zeros(nFrames, prod(win));
    
    for i = 1:nFrames
        try
            rect = [eyepos(i,[1 2]) win-1];
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

fprintf('Done\n')
%% concatenate data
n = (arrayfun(@(x) x.nFrames, trial));
trial(n == 0) = [];
nTrials = numel(trial);
n = [0; cumsum(n(n~=0))];

sacTimes = [];
sacSizes = [];
M = cell2mat(arrayfun(@(x) x.Mcrop, trial, 'uni', 0));

iix = all(M==0,2);
M(iix,:) = nan;

for iTrial = 1:nTrials
    sacTimes = [sacTimes; trial(iTrial).sacTimes + n(iTrial)];
    sacSizes = [sacSizes; trial(iTrial).sacSizes];
end

M = M ./ nanstd(M);
%% filter
figure(1); clf
subplot(1,2,1)
imagesc(ML(1:1e3,:))
subplot(1,2,2)
Mfilt = filter(TMagno, 1, M);
Tfilt = filter(TParvo, 1, M);
MfiltL = Mfilt*wtsL;
MfiltR = Mfilt*wtsR;
PfiltL = Tfilt*wtsL;
PfiltR = Tfilt*wtsR;

plot(MfiltL(1:10e3,:))


%%
NT = size(MfiltL,1);
EMagno = (MfiltL).^2 + (MfiltR).^2;
EParvo = (PfiltL).^2 + (PfiltR).^2;
figure(1); clf
E = EParvo;
fixdur = [diff(sacTimes); 0];
goodSacs = (fixdur > 200);
events = sacTimes(goodSacs);
for i = 1:nSfs
    [m, s, lags, wf] = eventTriggeredAverage(E(:,i)./nanmean(E(:,i)), events, [-50 400]);
    [~, ind] = sort(fixdur(goodSacs));
    subplot(2, nSfs,i)
    imagesc(wf(ind,:), [0 3])
    subplot(2, nSfs, i+nSfs)
    plot.errorbarFill(lags, m, s/sqrt(numel(events)));
    plot(lags, m); hold on
    plot(lags, s)
%     ylim([0 1])
    xlim([-50 200])
end



%%
figure(1); clf
goodSacs = ([diff(sacTimes); 0] > 200);
events = sacTimes(sacSizes<10 & goodSacs);
subplot(1,2,1)
nSfs = size(EMagno,2);
for i = 1:nSfs
    x = EMagno(:,i);
    [m, ~, lags] = eventTriggeredAverage(x, events, [-50 200]);
    plot(lags, m); hold on
end

subplot(1,2,2)
for i = 1:nSfs
    x = EParvo(:,i);
    [m, ~, lags] = eventTriggeredAverage(x, events, [-50 200]);
    plot(lags, m-mean(m)); hold on
end
%% plot
figure(1); clf
set(gcf, 'DefaultAxesColorOrder', lines)
ss = prctile(sacSizes, 0:25:100);
ss = [1 2 4 6 8];

% E = EMagno;
E = EParvo;
xwin = [-50 200];
cmap = parula(numel(ss));
for i = 1:nSfs
    subplot(1,nSfs,i)
    iix = (i-1)*prod(win) + (1:prod(win));
    for j = 1:numel(ss)-1
    
        ix = sacSizes > ss(j) & sacSizes < ss(j+1) & goodSacs;
        x = E(:,i);
        [m, ~, lags] = eventTriggeredAverage(x, sacTimes(ix), xwin);
        plot(lags, m, 'Color', cmap(j,:)); hold on
    end
    title(sfs(i))
    xlim(xwin)
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

