
%% import super sessions
% fdir =  '~/Google Drive/HuklabTreadmill/gratings/';
fpath = '~/Data/Datasets/HuklabTreadmill/';
fdir = '~/Data/Datasets/HuklabTreadmill/processed';

subjs = {'brie'};
for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    import_supersession(subj, fdir)
    fname = sprintf('%sD_all.mat', subj);
    movefile(fullfile(fdir, fname), fullfile(fpath, fname))
end
%%

fdir = '~/Data/Datasets/HuklabTreadmill/brain_observatory_1.1/';
subj = 'allen';
import_supersession(subj, fdir)

% copy files over
fpath = '~/Data/Datasets/HuklabTreadmill/';
fname = sprintf('%sD_all.mat', subj);
movefile(fullfile(fdir, fname), fullfile(fpath, fname))
%% plot RF centers
figdir = 'Figures/HuklabTreadmill/manuscript/';
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
subjs = {'gru', 'brie', 'allen'};
cmap = lines;
cmap = [cmap .1*ones(size(cmap,1), 1)];

figure(1); clf

for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    
    D = load_subject(subj, fpath);

    hasrf = find(~cellfun(@isempty, D.units));
    NC = numel(unique(D.spikeIds));
    
    fprintf('%s has %d/%d with RFs\n', subj, numel(hasrf), NC)

    x0 = cellfun(@(x) x{1}.center(1), D.units(hasrf));
    y0 = cellfun(@(x) x{1}.center(2), D.units(hasrf));
    mv = cellfun(@(x) x{1}.maxV, D.units(hasrf));
    iix = mv > 6 ;
    plot(x0(iix), y0(iix), '.', 'Color', cmap(isubj,:)); hold on
end

%% plot RF contours

for isubj = 1:numel(subjs)
    figure(isubj); clf

    subj = subjs{isubj};
    cmap = getcolormap(subj, false);

    D = load_subject(subj, fpath);

    hasrf = find(~cellfun(@isempty, D.units));
    NC = numel(unique(D.spikeIds));


    mv = cellfun(@(x) x{1}.maxV, D.units(hasrf));
    iix = mv > 10;

    hasrf = hasrf(iix);
    nrf = numel(hasrf);

    fprintf('%s has %d/%d with RFs\n', subj, nrf, NC)

    for i = 1:nrf
        rf = D.units{hasrf(i)}{1};
        if strcmp(subj, 'allen')
            rf.contour(:,1) = rf.contour(:,1) - 50;
        end
        rf.contour = rf.contour + randn*.2;
        plot(rf.contour(:,1), rf.contour(:,2), 'Color', [cmap(6,:) .5]); hold on
    end

    xlim([-50 50])
    ylim([-40 40])
    grid on

    plot(0, 0, '+k')
    plot.formatFig(gcf, [1.25 1], 'nature')
    set(gca, 'XTick', -50:10:50)
    set(gca, 'YTick', -40:10:40)
    saveas(gcf, fullfile(figdir, sprintf('rfs_%s.pdf', subj)))
end


%% plot subset of RFs
xd = [-50 50];
yd = [-40 40];
for isubj = 1:numel(subjs)
    figure(isubj); clf

    subj = subjs{isubj};
    cmap = getcolormap(subj, false);

    D = load_subject(subj, fpath);

    hasrf = find(~cellfun(@isempty, D.units));
    NC = numel(unique(D.spikeIds));


    mv = cellfun(@(x) x{1}.maxV, D.units(hasrf));
    iix = mv > 10;

    hasrf = hasrf(iix);

    rng(1)
    hasrf = randsample(hasrf, 50, false);
    nrf = numel(hasrf);

    fprintf('%s has %d/%d with RFs\n', subj, nrf, NC)
    
    plot(xd, [0 0], 'Color', [0 0 0 .3]);
    hold on
    plot([0 0], yd, 'Color', [0 0 0 .3]);

    for i = 1:nrf
        rf = D.units{hasrf(i)}{1};
        if strcmp(subj, 'allen')
            rf.contour(:,1) = rf.contour(:,1) - 50;
        end
        rf.contour = rf.contour + randn*.2;
        plot(rf.contour(:,1), rf.contour(:,2), 'Color', [cmap(6,:) .5]); hold on
    end

    xlim(xd)
    ylim(yd)

    plot(0, 0, '+k', 'MarkerSize', 2)

    plot(-40*[1 1], -30+[0 20], 'k', 'Linewidth', 2)
    plot(-40+[0 20], -30*[1 1], 'k', 'Linewidth', 2)

    plot.formatFig(gcf, [1.25 1], 'nature', 'OffsetAxesBool', true)
    
    set(gca, 'XTick', [xd(1) 0 xd(2)])
    set(gca, 'YTick', [yd(1) 0 yd(2)])
    xlim(xd)
    ylim(yd)

    axis equal
    saveas(gcf, fullfile(figdir, sprintf('rfs_subset_%s.pdf', subj)))
end

%%
circdiff = @(th1, th2) angle(exp(1i*(th1-th2)/180*pi))/pi*180;
winstart = 0.035;
winsize = 1;
prewin = .1;
run_thresh = 3;
debug = false;

subjs = {'gru', 'brie', 'allen'};
Stat = struct();
nsubjs = numel(subjs);
for isubj = 1:nsubjs
    subj = subjs{isubj};

    D = load_subject(subj, fpath);

    cids = unique(D.spikeIds);

    NC = numel(cids);
    baseline_subtract = true;

    Stat.(subj) = struct();
    Stat.(subj).pvis = nan(NC, 2);
    Stat.(subj).cid = nan(NC, 1);
    Stat.(subj).dsi = nan(NC, 3);
    Stat.(subj).gdsi = nan(NC, 3);
    Stat.(subj).osi = nan(NC, 3);
    Stat.(subj).gosi = nan(NC, 3);
    Stat.(subj).nori = nan(NC,3);
    Stat.(subj).ndir = nan(NC, 3);
    Stat.(subj).runmod = nan(NC,2);
    Stat.(subj).murun = nan(NC,2);
    Stat.(subj).mustat = nan(NC,2);
    Stat.(subj).prun = nan(NC,2);

    % get running speed during grating
    treadgood = find(~isnan(D.treadTime));
    [~, ~, id1] = histcounts(D.GratingOnsets, D.treadTime(treadgood));
    [~, ~, id2] = histcounts(D.GratingOffsets, D.treadTime(treadgood));
    bad = (id1 == 0 | id2 == 0);
    runningspeed = nan(numel(D.GratingOnsets), 1);
    id1 = treadgood(id1(~bad));
    id2 = treadgood(id2(~bad));
    runningspeed(~bad) = arrayfun(@(x,y) nanmean(D.treadSpeed(x:y)), id1, id2); %#ok<NANMEAN>
    

    total_directions = unique(D.GratingDirections(~isnan(D.GratingDirections)));
    total_orientations = unique(mod(total_directions, 180));
    num_directions = numel(total_directions);
    num_orientations = numel(total_orientations);
    
    Stat.(subj).meanrate = nan(NC,1);

    Stat.(subj).directions = total_directions;
    Stat.(subj).dratemarg = nan(NC,num_directions);
    Stat.(subj).dratebest = nan(NC,num_directions);
    Stat.(subj).drateweight = nan(NC,num_directions);

    Stat.(subj).orientations = total_orientations;
    Stat.(subj).oratemarg = nan(NC,num_orientations);
    Stat.(subj).oratebest = nan(NC,num_orientations);
    Stat.(subj).orateweight = nan(NC,num_orientations);

    Stat.(subj).robs = cell(NC,1);
    Stat.(subj).runningspeed = cell(NC,1);

    Stat.(subj).runrho = nan(NC,1);
    Stat.(subj).runrhop = nan(NC,1);

    for cc = 1:NC
        fprintf('%d/%d\n', cc, NC)
        cid = cids(cc);
        Stat.(subj).cid(cc) = cid;

        unitix = D.spikeIds == cid;
        sessix = unique(D.sessNumSpikes(unitix));

        gtix = find(ismember(D.sessNumGratings, sessix));
        gtix(isnan(D.GratingDirections(gtix))) = [];

        onsets = D.GratingOnsets(gtix);
        winsize = mode(D.GratingOffsets(gtix) - D.GratingOnsets(gtix));
        winsize = min(winsize, .5);

        t0 = min(onsets) - 2*winsize;
        st = D.spikeTimes(unitix) - t0;
        onsets = onsets - t0;

        st(st < min(onsets)) = [];
        sp = struct('st', st, 'clu', ones(numel(st),1));

        R = binNeuronSpikeTimesFast(sp, onsets + winstart, winsize);
        R0 = binNeuronSpikeTimesFast(sp, onsets - prewin, prewin);
        R = R ./ winsize;
        R0 = R0 ./ prewin;

        Stat.(subj).meanrate(cc) = mean(R);
        if Stat.(subj).meanrate(cc) < 1
            continue
        end
        
        if debug
            figure(1); clf
            subplot(2,1,1)
            plot(R, 'k'); hold on
            plot(xlim, mean(R0)*[1 1], 'r')
            pause
        end

        if baseline_subtract
            R = R - mean(R0);
        end

        % get conditions
        direction = D.GratingDirections(gtix);
        orientation = mod(direction, 180);
        speed = D.GratingSpeeds(gtix);
        freq = D.GratingFrequency(gtix);

        speeds = unique(speed(:))';
        freqs = unique(freq(:))';
        directions = unique(direction(:))';
        orientations = unique(orientation(:))';

        % bin all conditions
        Dmat = direction == directions;
        Omat = orientation == orientations;
        Fmat = freq == freqs;
        Smat = speed == speeds;

        nd = numel(directions);
        nf = numel(freqs);
        ns = numel(speeds);
        nt = numel(R);
        no = numel(orientations);
        Xbig = zeros(nt, nd, nf, ns);
        XbigO = zeros(nt, no, nf, ns);
        for iis = 1:ns
            for iif = 1:nf
                Xbig(:,:,iif,iis) = Dmat .* Fmat(:,iif) .* Smat(:,iis);
                XbigO(:,:,iif,iis) = Omat .* Fmat(:,iif) .* Smat(:,iis);
            end
        end

        runspeed = runningspeed(gtix);

        Stat.(subj).robs{cc} = R;
        Stat.(subj).runningspeed{cc} = runspeed;

        ix = ~isnan(runspeed);
        [rho, pval] = corr(R(ix), runspeed(ix), 'Type', 'Spearman');
        Stat.(subj).runrho(cc) = rho;
        Stat.(subj).runrhop(cc) = pval;

        statix = runspeed < run_thresh;
        runix = runspeed > run_thresh;

        Xbig = reshape(Xbig, nt, []);

        % get stimulus selectivity
        [i,group] = find(Xbig>0);
        [~, ind] = sort(i);
        group = group(ind);
    
        pval = anova1(R, group, 'off');
        Stat.(subj).pvis(cc,1) = pval;

        % get direction selectivity
        [i,group] = find(Dmat>0);
        [~, ind] = sort(i);
        group = group(ind);
    
        pval = anova1(R, group, 'off');
        Stat.(subj).pvis(cc,2) = pval;

        % direction marginalized
        x = Dmat.*R;
        mu = sum(x) ./ sum(Dmat);

        direc_ix = ismember(total_directions, directions);
        Stat.(subj).dratemarg(cc,direc_ix) = mu;
        Stat.(subj).gdsi(cc,1) = direction_selectivity_index(directions, mu, true);
        Stat.(subj).dsi(cc,1) = direction_selectivity_index(directions, mu, false);
        [~, mxid] = max(mu);
        Stat.(subj).ndir(cc,1) = sum(Dmat(:,mxid));

        Drun = Dmat(:,mxid).*runix;
        murun = sum(Drun.*R) ./ sum(Drun);
        Dstat = Dmat(:,mxid).*statix;
        mustat = sum(Dstat.*R) ./ sum(Dstat);
        C = sign(murun - mustat);
        rmax = max(murun, mustat);
        rmin = min(murun, mustat);
        Stat.(subj).runmod(cc,1) = C * (rmax - rmin) / abs(rmin);
        Stat.(subj).murun(cc,1) = murun;
        Stat.(subj).mustat(cc,1) = mustat;
        [~, Stat.(subj).prun(cc,1)] = ttest2(R(Drun>0), R(Dstat>0));

        % direction tuning at best SF / TF
        x = Xbig.*R;
        %     ci = bootci(100, @sum, x);
        mu = sum(x) ./ max(sum(Xbig),1);

        % get running modulation at best single stimulus
        [~, mxid] = max(mu);
        Drun = Xbig(:,mxid).*runix;
        murun = sum(Drun.*R) ./ sum(Drun);
        Dstat = Xbig(:,mxid).*statix;
        mustat = sum(Dstat.*R) ./ sum(Dstat);
        C = sign(murun - mustat);
        rmax = max(murun, mustat);
        rmin = min(murun, mustat);
        Stat.(subj).runmod(cc,2) = C * (rmax - rmin) / abs(rmin);
        Stat.(subj).murun(cc,2) = murun;
        Stat.(subj).mustat(cc,2) = mustat;
        [~, Stat.(subj).prun(cc,2)] = ttest2(R(Drun>0), R(Dstat>0));


        %     ci = ci ./ sum(Xbig);

        n = reshape(sum(Xbig), nd, []);
        I = reshape(mu, nd, []);
        [~, id] = max(sum(I));

        Stat.(subj).ndir(cc,2) = sum(n(:,id));

        mu = I(:,id)';
        Stat.(subj).dratebest(cc,direc_ix) = mu;
        Stat.(subj).gdsi(cc,2) = direction_selectivity_index(directions, mu, true);
        Stat.(subj).dsi(cc,2) = direction_selectivity_index(directions, mu, false);

        [~, mxid] = max(mu);
        Stat.(subj).ndir(cc,3) = n(mxid,id); % number of trials at best stimulus

        % direction tuning weighted by SF / TF tuning
        w = max(I)'; w = w ./ sum(w);
        X = reshape(reshape(Xbig, nt*nd, [])*w, nt, nd);
        x = X.*R;
        mu = sum(x) ./ max(sum(X),1);

        Stat.(subj).drateweight(cc,direc_ix) = mu;
        Stat.(subj).gdsi(cc,3) = direction_selectivity_index(directions, mu, true);
        Stat.(subj).dsi(cc,3) = direction_selectivity_index(directions, mu, false);


        % ORIENTATION
        Xbig = reshape(XbigO, nt, []);

        % orientation, marginalized
        x = Omat.*R;
        mu = sum(x) ./ sum(Omat);

        ori_ix = ismember(total_orientations, orientations);
        Stat.(subj).oratemarg(cc,ori_ix) = mu;
        Stat.(subj).gosi(cc,1) = orientation_selectivity_index(orientations, mu, true);
        Stat.(subj).osi(cc,1) = orientation_selectivity_index(orientations, mu, false);

        % orientation tuning, best SF / TF
        x = Xbig.*R;
        %     ci = bootci(100, @sum, x);
        mu = sum(x) ./ max(sum(Xbig),1);
        %     ci = ci ./ sum(Xbig);

        I = reshape(mu, no, []);
        [~, id] = max(sum(I));
        mu = I(:,id)';

        Stat.(subj).oratebest(cc,ori_ix) = mu;
        Stat.(subj).gosi(cc,2) = orientation_selectivity_index(orientations, mu, true);
        Stat.(subj).osi(cc,2) = orientation_selectivity_index(orientations, mu, false);

        % orientation tuning, weighted by SF / TF tuning
        w = max(I)'; w = w ./ sum(w);
        X = reshape(reshape(Xbig, nt*no, [])*w, nt, no);
        x = X.*R;
        mu = sum(x) ./ max(sum(X),1);

        Stat.(subj).orateweight(cc,ori_ix) = mu;
        Stat.(subj).gosi(cc,3) = orientation_selectivity_index(orientations, mu, true);
        Stat.(subj).osi(cc,3) = orientation_selectivity_index(orientations, mu, false);
    end

end

%%

subjs = fieldnames(Stat);
figure(1); clf
figure(2); clf


for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    
    good_ix = find(Stat.(subj).ndir(:,3) > 10 & max(Stat.(subj).dratebest,[],2) > 1);

    iix = Stat.(subj).pvis(good_ix) < 0.05;
    fprintf('Subject [%s] has %d /%d visually-selective units (%02.2f%%)\n', subj, sum(iix), numel(iix), mean(iix))

    figure(1)
    subplot(3,1,isubj)
    histogram(Stat.(subj).osi(good_ix(iix), 2), 'Normalization', 'pdf', 'binEdges', linspace(-2, 2, 100)); hold on
    xlabel('OSI')

    figure(2)
    subplot(3,1,isubj)
    histogram(Stat.(subj).gosi(good_ix(iix), 2), 'Normalization', 'pdf', 'binEdges', linspace(-2, 2, 100)); hold on
    xlabel('g OSI')
end

%% plot running correlation
nt = 400;
nsubjs = numel(subjs);

figure(1); clf

for isubj = 1:3
    subj = subjs{isubj};

    subplot(nsubjs, 1, isubj)
    numtrials = cellfun(@numel, Stat.(subj).robs);
    iix = numtrials > nt;
    n = sum(iix);
    h = histogram(Stat.(subj).runrho(iix), 'binEdges', linspace(-1, 1, 100), 'EdgeColor', 'none', 'FaceColor', repmat(.6, 1, 3)); hold on
    
    [~, pval, ci, tstat] = ttest(Stat.(subj).runrho(iix));
    mu = mean(Stat.(subj).runrho(iix));
    fprintf('%s mean rho %02.3f [%02.3f, %02.3f], p = %d, (t=%02.3f, df=%d)\n', subj, mu, ci(1), ci(2), pval, tstat.tstat, tstat.df)

    [pval, ~, tstat] = signrank(Stat.(subj).runrho(iix));
    mu = median(Stat.(subj).runrho(iix));
    ci = bootci(nboot, @median, Stat.(subj).runrho(iix));
    fprintf('%s median rho %02.3f [%02.3f, %02.3f], p = %d, (z=%02.3f, rank=%d)\n', subj, mu, ci(1), ci(2), pval, tstat.zval, tstat.signedrank)
    mx = max(h.Values);

    iix = iix & Stat.(subj).runrhop < 0.05;
    nsig = sum(iix);
    npos = sum(Stat.(subj).runrho(iix)>0);
    nneg = nsig - npos;

    histogram(Stat.(subj).runrho(iix), 'binEdges', linspace(-1, 1, 100), 'EdgeColor', 'none', 'FaceColor', repmat(.1, 1, 3));
    
    plot(ci, [1 1]*mx*1.05, '-', 'Color', repmat(.6, 1, 3), 'Linewidth', 2)
    plot(mu, mx*1.05, 'o', 'MarkerFaceColor', repmat(.1, 1, 3), 'Color',repmat(.1, 1, 3), 'MarkerSize', 2)
    
    ylim([0 1.1*mx])
    xlim([-1 1])
    fprintf('%s: %d/%d\n', subj, nsig, n)
    fprintf('%d positive %d negative (ratio = %02.2f)\n', npos, nneg, npos./nneg)
end

xlabel("Spearman's rho")
plot.formatFig(gcf, [1 2], 'nature')
saveas(gcf, fullfile(figdir, 'runcorrhists.pdf'))
%% decode stimulus
circdiff = @(x,y) angle(exp(1i*(x - y)/180*pi))/pi*180;

% Dstat = struct();
for isubj = 3%1:nsubjs
    subj = subjs{isubj};

    D = load_subject(subj, fpath);
    sess_ids = unique(D.sessNumGratings);


    
    Nsess = numel(sess_ids);
    Dstat.(subj).decoding = cell(Nsess,1);

    for isess = 1:Nsess
        Dstat.(subj).decoding{isess} = decode_stim(D, sess_ids(isess), 'runThreshold', 3);
    end
end

%%
mR = zeros(Nsess,1);
mRCi = zeros(Nsess,2);
mS = zeros(Nsess,1);
mSCi = zeros(Nsess,2);
nS = zeros(Nsess,1);
nR = zeros(Nsess,1);

figure(1); clf
for iSess = 1:Nsess
    aerr = abs(circdiff(Dstat(iSess).Stim, Dstat(iSess).decoderStimTot));
    
    inds = Dstat(iSess).runTrials;
    nR(iSess) = numel(inds);
    mR(iSess) = median(aerr(inds));
    mRCi(iSess,:) = bootci(1000, @median, aerr(inds));
    
    inds = setdiff(1:Dstat(iSess).NTrials, Dstat(iSess).runTrials);
    nS(iSess) = numel(inds);
    mS(iSess) = median(aerr(inds));
    mSCi(iSess,:) = bootci(1000, @median, aerr(inds));
    
    plot(mS(iSess)*[1 1], mRCi(iSess,:), 'Color', .5*[1 1 1]); hold on
    plot(mSCi(iSess,:), mR(iSess)*[1 1], 'Color', .5*[1 1 1]);
    h = plot(mS(iSess), mR(iSess), 'o');
    h.MarkerFaceColor = h.Color;
    
end

plot(xlim, xlim, 'k')
xlim([0 20])
ylim([0 20])
xlabel('Stationary')
ylabel('Running')
title('Median Decoding Error (degrees)')

%%



cc = cc + 1;

subj = 'gru';
if cc > size(Stat.(subj).pvis,1)
    cc = 1;
end
figure(1); clf
plot(Stat.(subj).directions, Stat.(subj).dratebest(cc,:))
Stat.(subj).pvis(cc,:)

%%
figure(2); clf
figure(3); clf
for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    good_ix = find(Stat.(subj).ndir(:,2) > 100 & Stat.(subj).meanrate > 1);
    disp(subj)
    figure(2)
    subplot(3,1,isubj)
    histogram(Stat.(subj).prun(good_ix,2), linspace(0, 1, 100)); hold on
    

    mean(Stat.(subj).prun(good_ix,:) < 0.05)

    sigix = Stat.(subj).prun(good_ix,:) < 0.05;

    figure(3);
    subplot(3,1,isubj)
    h = histogram(Stat.(subj).runmod(good_ix,2), 'binEdges', linspace(-10, 10, 150), 'FaceColor', repmat(.5, 1, 3)); hold on
    h1 = histogram(Stat.(subj).runmod(good_ix(sigix(:,2)),2), 'binEdges', linspace(-10, 10, 150), 'FaceColor', 'c'); hold on
    w = h.Values / sum(h.Values);
    bc = h.BinEdges(1:end-1) + diff(h.BinEdges)/2;
    com = bc*w(:);
    plot(com, max(h.Values), 'vr')
    text(4, max(h.Values)*.8, sprintf('n = %d', numel(good_ix)))

end


%%
subj = 'allen';
D = load_subject(subj, fpath);

%%
cids = unique(D.spikeIds);
NC = numel(cids);
assert(NC == size(Stat.(subj).prun,1))

good_ix = find(Stat.(subj).ndir(:,2) > 100 & Stat.(subj).meanrate > 1);

sigix = find(Stat.(subj).prun(good_ix,2) < 0.05);
fprintf('%s has %d significant running modulation\n', subj, numel(sigix))

[~, ind] = sort(Stat.(subj).runmod(sigix,2), 'ascend');
sigix = sigix(ind);
%%
cc = cc + 1;
if cc > numel(sigix)
    cc = 1;
end

cid = cids(good_ix(sigix(cc)));
[stim, robs, behavior, opts] = bin_ssunit(D, cid, 'plot', false);


tspeed = nanmean(behavior{1},2); %#ok<*NANMEAN> 
r = mean(robs(:,opts.lags > 0 & opts.lags < 2),2);

figure(1); clf
plot(r, 'k'); hold on
plot(tspeed / max(tspeed) * max(r) , 'c')

figure(2); clf
iistat = tspeed < 3;
histogram(r(iistat)); hold on
histogram(r(tspeed > 3))

%%

figure(1); clf
bs = mean(diff(opts.lags));
r = reshape(robs', [], 1);
tread = reshape(behavior{1}', [], 1);
tread = imboxfilt(tread, 501);
k = exp(-.5*((0:50)/10^2).^2); k = k ./ sum(k);
% r = filter(k,1,r);
r = imboxfilt(r, 501);

iix = ~isnan(tread);
plot(r(iix), 'k')
hold on
plot(tread(iix)/max(tread(iix))*max(r), 'c')
title(cc)
% behavior{1}'

%%
[~, ind] = sort(stim);
figure(1); clf
imagesc(robs(ind,:))





