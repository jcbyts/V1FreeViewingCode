
%%


% --- get path
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
subj = 'gru';

% --- get concatenated spikes file
spikesfname = fullfile(fpath, subj, [subj '_All_cat.mat']);
fprintf('Loading spikes from [%s]\n', spikesfname)
EphysData = load(spikesfname);

% --- find ephys and behavioral date tags
ephysSessions = EphysData.z.RecId';
behaveSessions = io.dataFactoryTreadmillSS';

% --- convert dates to number and matsch sessions
behaveDates = cellfun(@(x) datenum(x(numel(subj)+2:end), 'yyyymmdd'), behaveSessions);
ephysDates = cellfun(@(x) datenum(x(numel(subj)+2:numel(subj)+11), 'yyyy-mm-dd'), ephysSessions);

[sessionMatch, behave2ephysNum] = ismember(ephysDates, behaveDates);

goodEphysSessions = find(sessionMatch);

%% --- loop over sessions, build big spike struct

Dbig = struct('GratingDirections', [], 'GratingFrequency', [], ...
    'GratingOffsets', [], 'GratingOnsets', [], 'GratingSpeeds', [], ...
    'eyeLabels', [], 'eyePos', [], 'eyeTime', [], ...
    'treadSpeed', [], 'treadTime', [], ...
    'sessNum', [], 'spikeTimes', [], 'spikeIds', []);

fprintf('Loading and aligning spikes with behavior\n')

startTime = 0; % all experiments start at zero. this number will increment as we concatenate sessions
newOffset = 0;
timingFields = {'GratingOffsets', 'GratingOffsets', 'spikeTimes', 'treadTime', 'eyeTime'};
nonTimingFields = {'GratingDirections', 'GratingFrequency', 'GratingSpeeds', 'eyeLabels', 'eyePos', 'treadSpeed', 'spikeIds', 'sessNum'};

for iSess = 1:numel(goodEphysSessions)
    
    rId = goodEphysSessions(iSess);
    bSess = behave2ephysNum(goodEphysSessions(iSess));
    
    fprintf('Session %d/%d [ephys:%d , behav:%d]\n', iSess, numel(goodEphysSessions), rId, bSess)
    
    unitlist = find(cellfun(@(x) ~isempty(x), EphysData.z.Times{rId}));

    st = [];
    clu = [];

    for iunit = 1:numel(unitlist)
        kunit = unitlist(iunit);
    
        stmp = double(EphysData.z.Times{rId}{kunit}) / EphysData.z.Sampling;
        st = [st; stmp];
        clu = [clu; kunit*ones(numel(stmp),1)];
    end
    
    D_ = io.dataFactoryTreadmillSS(bSess);
    D_.spikeTimes = st;
    D_.spikeIds = clu;
    
    % convert to D struct format
    D = io.get_drifting_grating_output(D_);
    D.sessNum = iSess*ones(size(st));
    
    % loop over timing fields and offset time
    for iField = 1:numel(timingFields)
        Dbig.(timingFields{iField}) = [Dbig.(timingFields{iField}); D.(timingFields{iField}) + startTime];
        newOffset = max(newOffset, max(Dbig.(timingFields{iField}))); % track the end of this session
    end
    
    for iField = 1:numel(nonTimingFields)
        Dbig.(nonTimingFields{iField}) = [Dbig.(nonTimingFields{iField}); D.(nonTimingFields{iField})];
    end
    
    startTime = newOffset + 2; % 2 seconds between sessions
        
end




%%


st = [];
clu = [];

D = io.get_drifting_grating_output(D_);

%%

id = strfind(S.processedFileName,'_');
subj = S.processedFileName(1:id);
dat = datenum(S.processedFileName(id+(1:8)), 'yyyymmdd');
dateStr = [subj datestr(dat, 'yyyy-mm-dd')];
st = [];
clu = [];

recId = find(contains(EphysData.z.RecId, dateStr));
for rId = recId(:)'
    
    unitlist = find(cellfun(@(x) ~isempty(x), EphysData.z.Times{rId}));
    
    for iunit = 1:numel(unitlist)
        kunit = unitlist(iunit);
        
        stmp = double(EphysData.z.Times{rId}{kunit}) / EphysData.z.Sampling;
        st = [st; stmp];
        clu = [clu; iunit*ones(numel(stmp),1)];
    end
    
end

% Exp.osp = struct('st', st, 'clu', clu, 'cids', 1:numel(unitlist));
% fprintf('Done\n')

%%

[Exp,S] = io.dataFactoryTreadmill(4);
% add unit quality (since some analyses require this field)
Exp.osp.cgs = ones(size(Exp.osp.cids))*2;
io.checkCalibration(Exp);

%% Forward correlation spatial RF mapping
BIGROI = [-10 -8 10 8];

stat = spat_rf_helper(Exp, 'ROI', BIGROI, ...
    'win', [0 10],...
    'stimList', {'Dots'}, ...
    'binSize', .3, 'plot', true, 'debug', true, 'boxfilt', 9, 'spikesmooth', 1);

%% saccade modulation
frf = fixrate_by_stim(Exp, 'stimSets', {'DriftingGrating', 'BackImage'});

%% Grating tuning
EphysData = io.get_drifting_grating_output(Exp);

%% bin spikes in window aligned ton onset
binsize = 1e-3;
win = [-.25 1.1];

rem = find(EphysData.GratingOnsets+win(2) > max(EphysData.spikeTimes), 1);
EphysData.GratingOnsets(rem:end) = [];
EphysData.GratingOffsets(rem:end) = [];
EphysData.GratingDirections(rem:end) = [];
EphysData.GratingFrequency(rem:end) = [];
EphysData.GratingSpeeds(rem:end) = [];


NC = max(EphysData.spikeIds);
bs = (EphysData.spikeTimes==0) + ceil(EphysData.spikeTimes/binsize);

spbn = sparse(bs, EphysData.spikeIds, ones(numel(bs), 1));
lags = win(1):binsize:win(2);
blags = ceil(lags/binsize);

numlags = numel(blags);
balign = ceil(EphysData.GratingOnsets/binsize);
numStim = numel(balign);

% binning treadmill speed
treadBins = ceil(EphysData.treadTime/binsize);
treadSpd = nan(size(spbn,1), 1);
treadSpd(treadBins(~isnan(EphysData.treadSpeed))) = EphysData.treadSpeed(~isnan(EphysData.treadSpeed));
treadSpd = max(repnan(treadSpd, 'v5cubic'),0);

% Do the binning here
disp('Binning spikes')
spks = zeros(numStim, NC, numlags);
tspd = zeros(numStim, numlags);
for i = 1:numlags
    spks(:,:,i) = spbn(balign + blags(i),:);
    tspd(:,i) = treadSpd(balign + blags(i));
end

disp('Done')
cc = 1;


%% plot distribution of running speeds
thresh = 1;

figure(10); clf
runningSpd = mean(tspd,2);
runningTrial = runningSpd > thresh;

histogram(runningSpd(~runningTrial), 'binEdges', 0:.1:25, 'FaceColor', [.5 .5 .5])
hold on
clr = [1 .2 .2];
histogram(runningSpd(runningTrial), 'binEdges', 0:.1:25, 'FaceColor', clr)
plot.fixfigure(gcf,10,[4 4]);
xlabel('Running Speed (units?)')
ylabel('Trial Count') 
text(.1, mean(ylim), sprintf('Running Trials (n=%d)', sum(runningTrial)), 'Color', clr, 'FontSize', 14)
set(gca, 'YScale', 'linear')

%%
% cc = 1
ths = unique(EphysData.GratingDirections);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(1); clf
% ax = plot.tight_subplot(sx, sy, 0.01, 0.01);
% 
% for cc = 1:NC
%     set(gcf, 'currentaxes', ax(cc))
    
cc = cc + 1;
if cc > NC
    cc = 1;
end


subplot(1,2,1) % no running

thctr = 0;
for th = 1:numel(ths)
    iix = find(EphysData.GratingDirections==ths(th) & ~runningTrial);
%     sum(
    nt = numel(iix);
    spk = squeeze(spks(iix,cc,:));
    [ii,jj] = find(spk);
    plot.raster(lags(jj), ii+thctr, 2, 'Color', 'k' ); hold on
    thctr = thctr + nt;
end
title('No Running')
axis tight

subplot(1,2,2)

thctr = 0;
for th = 1:numel(ths)
    iix = find(EphysData.GratingDirections==ths(th) & runningTrial);
%     sum(
    nt = numel(iix);
    spk = squeeze(spks(iix,cc,:));
    [ii,jj] = find(spk);
    plot.raster(lags(jj), ii+thctr, 2, 'Color', 'k' ); hold on
    thctr = thctr + nt;
end
title('Running')
axis tight

% end



%% Get PSTHs
fkern = ones(20,1)/20;
spksfilt = spks;
for i = 1:numStim
    spksfilt(i,:,:) = filtfilt(fkern, 1, squeeze(spksfilt(i,:,:))')';
end

nd = numel(ths);
psthsRunning = zeros(numlags, nd, NC);
psthsNoRunning = zeros(numlags, nd, NC);
for i = 1:nd
    iix = EphysData.GratingDirections==ths(i);
    psthsRunning(:,i,:) = squeeze(mean(spksfilt(iix & runningTrial,:,:),1))';
    psthsNoRunning(:,i,:) = squeeze(mean(spksfilt(iix & ~runningTrial,:,:),1))';
end

% trim filtering artifacts
psthsRunning(1:10,:,:) = nan;
psthsRunning(end-10:end,:,:) = nan;
psthsNoRunning(1:10,:,:) = nan;
psthsNoRunning(end-10:end,:,:) = nan;

%% Step over cells, plot PSTH as image
cc = cc + 1;
if cc > NC
    cc = 1;
end
vals = [reshape(psthsRunning(:,:,cc), [], 1); reshape(psthsNoRunning(:,:,cc), [], 1)];
clim = [min(vals(:)) max(vals(:))];
figure(1); clf
subplot(1,2,1)

imagesc(lags, ths, psthsRunning(:,:,cc)', clim)
xlabel('Direction')
ylabel('Time')
title('Running')

subplot(1,2,2)
imagesc(lags, ths, psthsNoRunning(:,:,cc)', clim)
xlabel('Direction')
ylabel('Time')
title('No Running')
colormap(plot.viridis)


%% plot Tuning Curves
win = [0.04 1.2];
iix = lags > win(1) & lags < win(2);
tdur = lags(find(iix,1,'last'))-lags(find(iix,1,'first'));
tcRun = squeeze(nansum(psthsRunning(iix,:,:)))/tdur;
tcNoRun = squeeze(nansum(psthsNoRunning(iix,:,:)))/tdur;
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

figure(1); clf
for cc = 1:NC
    subplot(sx, sy, cc)
    plot(ths, tcRun(:,cc), 'k', 'Linewidth', 2); hold on
    plot(ths, tcNoRun(:,cc), 'r', 'Linewidth', 2); hold on
    set(gca, 'XTick', 0:180:360)
    xlim([0 360])
    title(cc)
end

plot.fixfigure(gcf, 12, [14 14])
a = plot.suplabel('Spike Rate', 'y'); 
a.FontSize = 20;
a = plot.suplabel('Direction', 'x');
a.FontSize = 20;

%%
figure(2); clf
set(gcf, 'Color', 'w')
plot(max(tcNoRun), max(tcRun), 'ow', 'MarkerFaceColor', .5*[1 1 1])
hold on
plot(xlim, xlim, 'k')
xlabel('Max Rate (Stationary)')
ylabel('Max Rate (Running)')
%%
cc = 1;

%%

cc = cc + 1;
if cc > NC
    cc = 1;
end

figure(10); clf
nth = numel(unique(EphysData.GratingDirections));
cmap = parula(nth);
ax = plot.tight_subplot(2, nth, 0.01, 0.01);
    
vals = [reshape(psthsRunning(:,:,cc), [], 1); reshape(psthsNoRunning(:,:,cc), [], 1)]/1e-3;
clim = [min(vals(:)) max(vals(:))];


for ith = 1:nth
    
    set(gcf, 'currentAxes', ax(ith));
    plot(lags, imgaussfilt(psthsNoRunning(:,ith,cc)/1e-3, 25), 'Color', cmap(ith,:), 'Linewidth', 2); hold on
    clr = (cmap(ith,:) + [1 1 1])/2;
    plot(lags, imgaussfilt(psthsRunning(:,ith,cc)/1e-3, 25), '-', 'Color', clr, 'Linewidth', 2); hold on
    ylim(clim)
    axis off
    if ith==1
        text(lags(1), .9*clim(2), sprintf('Unit: %d', cc))
        text(lags(1), .8*clim(2), 'Running', 'Color', clr)
        text(lags(1), .7*clim(2), 'No Running', 'Color', cmap(ith,:))
    end
    set(gcf, 'currentAxes', ax(ith+nth));
    [dx, dy] = pol2cart(ths(ith)/180*pi, 1);
    q = quiver(0,0,dx,dy,'Color', cmap(ith,:), 'Linewidth', 5, 'MaxHeadSize', 2); hold on
%     plot([0 dx], [0 dy], 'Color', cmap(ith,:), 'Linewidth', 5); hold on
%     plot(dx, dy, 'o', 'Color', cmap(ith,:), 'MarkerFaceColor', cmap(ith,:), 'MarkerSize', 20)
    
%     R = [cos(pi/2) sin(pi/2); -sin(pi/2) -cos(pi/2)];
    
    
%     for i = [90 270]
%         [dx, dy] = pol2cart((ths(ith) + i)/180*pi, .1);
%         plot(-dx, -dy, 'o', 'Color', cmap(ith,:), 'MarkerFaceColor', cmap(ith,:), 'MarkerSize', 10)
% %     S = [1 0; 0 1];
% %     dxdy = [dx dy] * R*S;
% %     plot(dxdy(1), dxdy(2), 
% %     dxdy = [dx dy] * -R*S;
% %     plot(dxdy(1), dxdy(2), 'o', 'Color', cmap(ith,:), 'MarkerFaceColor', cmap(ith,:), 'MarkerSize', 20)
%     
%     end
    xlim([-1 1]*2)
    ylim([-1 1]*2)
    axis off
end

set(gcf, 'Color', 'w')


%%

[Exp,S] = io.dataFactoryTreadmill(6);
% add unit quality (since some analyses require this field)
Exp.osp.cgs = ones(size(Exp.osp.cids))*2;
io.checkCalibration(Exp);

EphysData = io.get_drifting_grating_output(Exp);

exname = Exp.FileTag;
outdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'processed');
fname = fullfile(outdir, exname);

save(fname, '-v7', '-struct', 'D')

%% copy to server (for python analyses)
old_dir = pwd;

cd(outdir)
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/HuklabTreadmill/processed/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command exname ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)

cd(old_dir)


