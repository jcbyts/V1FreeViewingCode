
%% Setup paths
fdir = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');

flist = dir(fullfile(fdir, '*.mat'));
figdir = 'Figures/HuklabTreadmill/manuscript/';

subjs = {'gru', 'brie','allen'};
dirname = {'gratings', 'gratings', 'brain_observatory_1.1'};

S = struct();


%% Run analysis
for isubj = 1:numel(subjs)
    
    subj = subjs{isubj};

    flist = dir(fullfile(fdir, dirname{isubj}, [subj '*.mat']));

    nsess = numel(flist);

    S.(subj) = struct();
    S.(subj).rpc = {};
    S.(subj).runningspeed = {};
    S.(subj).rhounit = {};
    S.(subj).pvalunit = {};
    S.(subj).robs = {};
    S.(subj).cids = {};
    S.(subj).rho = nan(nsess,1);
    S.(subj).pval = nan(nsess,1);
    S.(subj).sessname = {flist(:).name};

    fprintf('%s has %d sessions\n', subj, nsess)
    for isess = 1:nsess
        
        
        D = load(fullfile(flist(isess).folder, flist(isess).name));
        if isempty(D.GratingOnsets)
            continue % why is this here!
        end

        D = fix_scipy_weirdness(D);
        % main analysis code goes here
        [tmp, opts] = running_vs_spikePC(D, opts);

        S.(subj).rpc{isess} = tmp.rpc;
        S.(subj).runningspeed{isess} = tmp.runningspeed;
        S.(subj).rho(isess) = tmp.rho;
        S.(subj).pval(isess) = tmp.pval;
        S.(subj).rhounit{isess} = tmp.rhounit;
        S.(subj).pvalunit{isess} = tmp.pvalunit;
        S.(subj).robs{isess} = tmp.robs;
        S.(subj).cids{isess} = tmp.cids;
    end

end

%% plot summary of session level PCs
nt = 400;

figure(2); clf
subjs = fieldnames(S);
nsubjs = numel(subjs);
for isubj = 1:nsubjs
    subj = subjs{isubj};
    sessix = ~isnan(S.(subj).rho) & cellfun(@numel, S.(subj).runningspeed(:)) > nt;
    subplot(nsubjs, 1, isubj)
    histogram(S.(subj).rho(sessix), 'binEdges', linspace(-1, 1, 50), 'FaceColor', repmat(.6, 1, 3)); hold on

    ix = sessix & S.(subj).pval < 0.05;
    histogram(S.(subj).rho(ix), 'binEdges', linspace(-1, 1, 50), 'FaceColor', repmat(.1, 1, 3)); hold on
    title(subj)
end

%% plot the most correlated sessions

for isubj = 1:nsubjs
    figure(isubj); clf
    subj = subjs{isubj};
    sessix = find(~isnan(S.(subj).rho));
    sessix = sessix(cellfun(@numel, S.(subj).runningspeed(sessix)) > nt);

    [~, ind] = sort(S.(subj).rho(sessix), 'descend');

    rhos = S.(subj).rho(sessix(ind));

    n = 5;
    for i = 1:n
        subplot(n, 1, i)
        id = sessix(ind(i));
        yyaxis left
        plot(S.(subj).rpc{id}(:,1), 'k');
        ylim([-1 1])
        xlim([1 nt])
        yyaxis right
        plot(S.(subj).runningspeed{id}, 'Color', repmat(.5, 1, 3)); hold on
        xlim([1 nt])
        ylim([-5 1.5*max(S.(subj).runningspeed{id})])
        if i==1
            title(subj)
        end
    end
    xlabel('Trial #')
end


%% plot the most correlated session for each subject and the spike counts sorted by correlation coefficient

nt = 400;

for isubj = 1:nsubjs
    subj = subjs{isubj};
    sessix = find(~isnan(S.(subj).rho));
    sessix = sessix(cellfun(@numel, S.(subj).runningspeed(sessix)) > nt);

    [~, ind] = sort(S.(subj).rho(sessix), 'descend');

    rhos = S.(subj).rho(sessix(ind));

    n = numel(sessix);
    for i = 1:n
        figure((isubj-1)*n + i); clf
        set(gcf, 'Color', 'w')

        id = sessix(ind(i));
        [~, rhoind] = sort(S.(subj).rhounit{id});
        r = S.(subj).robs{id}(:,rhoind);
        r = (r - min(r)) ./ range(r); % normalize

        subplot(3,1,1:2)
        imagesc(r'); colormap(1-gray)

        xlim([1 nt])
        title(sprintf('%s %d', subj, id))

        subplot(3,1,3)
        yyaxis left
        c = [0 0 0];
        plot(S.(subj).rpc{id}(:,1), 'Color', c);
        set(gca, 'YColor', c)

        ylim([-1 1])
        xlim([1 nt])
        yyaxis right
        c = repmat(.5, 1, 3);
        plot(S.(subj).runningspeed{id}, 'Color', c); hold on
        set(gca, 'YColor', c)
        ylabel('Running Speed (cm/s)')
        xlim([1 nt])
        ylim([-5 1.5*max(S.(subj).runningspeed{id})])
        %     if i==1
        %         title(subj)
        %     end
        plot.formatFig(gcf, [6 3], 'nature')
        saveas(gcf, fullfile(figdir, sprintf('runpc_%s_%d_%d.pdf', subj, i, id)))
    end
    xlabel('Trial #')
end

%% get all units and 
nboot = 100;

sessnum = [];
unitid = [];
rhounit = [];
pvalunit = [];
subjnum = [];

figure(4); clf
for isubj = 1:nsubjs

    subj = subjs{isubj};
    subplot(nsubjs, 1, isubj)

    sessix = ~isnan(S.(subj).rho) & cellfun(@numel, S.(subj).runningspeed(:)) > nt;
    for isess = find(sessix(:))'
        
        rho = S.(subj).rhounit{isess};
        NC = numel(rho);
        pval = S.(subj).pvalunit{isess};

        sessnum = [sessnum; isess*ones(NC,1)];
        unitid = [unitid; S.(subj).cids{isess}];
        rhounit = [rhounit; rho(:)];
        pvalunit = [pvalunit; pval(:)];
        subjnum = [subjnum; isubj*ones(NC, 1)];
    end

    iix = subjnum==isubj;
    h = histogram(rhounit(iix), 'binEdges', linspace(-1, 1, 100), 'FaceColor', repmat(.6, 1, 3)); hold on
    
    mu = mean(rhounit(iix));
    [~, pval, ci, stats] = ttest(rhounit(iix));



    iix = iix & pvalunit < 0.05;
    histogram(rhounit(iix), 'binEdges', linspace(-1, 1, 100), 'FaceColor', repmat(.1, 1, 3)); hold on
    title(subj)
end

xlabel('Spearman rank correlation')
%%

for isubj = 1:nsubjs
    subj = subjs{isubj};

    figure(isubj); clf

    iix= subjnum == isubj;
    iix = iix & pvalunit < 0.05;

    [rhosort, ind] = sort(rhounit(iix), 'descend');

    sess = sessnum(iix);
    sess = sess(ind);

    unit = unitid(iix);
    unit = unit(ind);

    for i = 1:n
        assert(S.(subj).rhounit{sess(i)}(unit(i))==rhosort(i), 'rho does not match')

        subplot(n, 1, i)
        yyaxis left
        r = S.(subj).robs{sess(i)}(:,unit(i));
        plot(r, 'k');
        ylim([-.5*max(r) max(r)])
        xlim([1 nt])
        yyaxis right
        plot(S.(subj).runningspeed{sess(i)}, 'Color', repmat(.5, 1, 3)); hold on
        xlim([1 nt])
        ylim([-5 1.5*max(S.(subj).runningspeed{sess(i)})])
        if i==1
            title(subj)
        end
    end
    xlabel('Trial #')

end

%%



