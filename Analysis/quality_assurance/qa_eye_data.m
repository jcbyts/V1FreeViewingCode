
%% load session
Exp = io.dataFactoryGratingSubspace(15);

%% plot raw eye traces
close all
tt = Exp.vpx2ephys(Exp.vpx.smo(:,1));
ex = Exp.vpx.smo(:,2);
ey = Exp.vpx.smo(:,3);

figure(1); clf
plot(tt, ex, tt, ey)
ylim([-15 15])


%% plot spikes in time
tic
rTotal = binNeuronSpikeTimesFast(Exp.osp, tt, .1);
toc
%%
figure(1); clf
Rsmo = imgaussfilt(sum(rTotal,2), 5);
plot(tt, Rsmo)

%%
amp = hypot(ex,ey);
spd = Exp.vpx.smo(:,7)/200;

amp(amp > 20) = nan;

figure(1); clf
plot(tt, amp)

figure(2); clf
plot(tt, spd); hold on
plot(tt, Rsmo)
%%
% [xc, bc] = xcorr(spd, Rsmo, 100, 'unbiased');
figure(1); clf


ev = Exp.slist(:,4);
win = [-200 200];
[an, sd, bc, wfs] = eventTriggeredAverage(Rsmo, ev, win);

plot(bc, an);

%% load old processed data from perisaccadic RF

addpath ~/Dropbox/MatlabCode/Repos/pdstools/
cc = 0;
D = load('/Users/jcbyts/Dropbox/Projects/perisaccadicRF/data/v1/e_20170824_v1_2d23.mat');
Exp = io.dataFactoryGratingSubspace(18);
%% sort events
% ev = D.saccades.tstart;
% ev = saccades.tstart;
ev = Exp.vpx2ephys(Exp.slist(:,1));
eventField = 'hartleyFF';
validIx = getTimeIdx(ev(:,1), [D.(eventField).start], [D.(eventField).stop]);
ev = ev(validIx,:);

win = [-.5 .5];
bs = 8e-3;
[~,~,~,~,sev] = pdsa.eventPsth(ev,ev,win, bs);

[~,zb] = min(bc.^2);
n = numel(ev);
pre = nan(n,1);
post = nan(n,1);
for i = 1:n
    if sum(sev(i,1:zb-1))~=0
        pre(i) = find(sev(i,1:zb-1), 1, 'last');
    end
    
    if sum(sev(i,zb:end))>0
        post(i) = find(sev(i,zb:end), 1, 'first');
    end
end


cc = 0;
%%
figure(1); clf
cc = 1; %cc + 1;

cids = D.spikes.cids;
st1 = D.spikes.st(D.spikes.clu==cids(cc));

[~, ind] = sort(Exp.osp.clusterDepths);
cids2 = Exp.osp.cids(ind);
st2 = Exp.osp.st(Exp.osp.clu==cids2(cc));


[m,~,bc, ~,wfs] = pdsa.eventPsth(st1, ev, win, bs);
[m2,~,bc2, ~,wfs2] = pdsa.eventPsth(st2, ev, win, bs);

% imagesc(wfs)
plot(bc, m); hold on
plot(bc2, m2, '--')
title(cc)

figure(2); clf
[~, ind] = sort(pre);

subplot(1,3,1)
imagesc(bc, 1:numel(ind), sev(ind,:))
ylim([1 3500])
title('Saccades Aligned to Saccades')
xlabel('Time from saccade onset')
ylabel('Saccade Number')

subplot(1,3,2)
[i,j] = find(wfs(ind,:));
plot.raster(j,i,5); axis ij
ylim([1 3500])
axis tight
title('Spikes Aligned to Saccades')
xlabel('Time from saccade onset')

subplot(1,3,3)
[i,j] = find(wfs2(ind,:));
plot.raster(j,i,5); axis ij
ylim([1 3500])

%%
amp = hypot(Exp.vpx.smo(:,2),Exp.vpx.smo(:,3));
goodix = find(amp < 20);
saccades = findSaccades(Exp.vpx.smo(goodix,1),Exp.vpx.smo(goodix,2),Exp.vpx.smo(goodix,3),...
    'accthresh', 2e4,...
    'velthresh', 10,...
    'velpeak', 10,...
    'isi', 0.04);

slist = [saccades.tstart, saccades.tend, saccades.tpeak, ...
    goodix(saccades.startIndex), goodix(saccades.endIndex), goodix(saccades.peakIndex)];


%   accthresh - acceleration threshold (default: 2e4 deg./s^2)
%   velthresh - velocity threshold (default: 10 deg./s)
%   velpeak   - minimum peak velocity (default: 10 deg./s)
%   isi       - minimum inter-saccade interval (default: 0.050s)



