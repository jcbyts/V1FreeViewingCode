function do_regression_analysis_supersession()
% [Stim, opts, Rpred, Running] = do_regression_analysis(D)

%% Get all relevant covariates
fdir = '~/Data/Datasets/HuklabTreadmill/';
fout = '~/Data/Datasets/HuklabTreadmill/regression_ss/';

%% loop over subjects and run super session analysis
subjs = {'gru', 'brie'};
for isubj = 1:numel(subjs)
    
    subj = subjs{isubj};
    fprintf('***************\n\n***************\n\n')
    fprintf('Analyzing subject [%s]\n', subj)

    D = load_subject(subj, fdir);
    Stim = convert_Dstruct_to_trials(D, 'bin_size', 1/60, 'pre_stim', .2, 'post_stim', .2);
    
    cids = unique(D.spikeIds);
    NC = numel(cids);
    
    D.subj = subj;
    
    parfor cc = 1:NC
    
        do_regression_ss(Stim, D, cids(cc), fout);
    
    end
end

%%

subj = 'brie';

flist = dir(fullfile(fout, [subj '*']));

Rpred = struct();
models2get = {'stimsac', 'stimrunsac', 'drift', 'RunningGain', 'PupilGain'};
for imodel = 1:numel(models2get)
    Rpred.(models2get{imodel}).r2 = [];
end

for cc = 1:numel(flist)
    load(fullfile(flist(cc).folder, flist(cc).name), 'Rpred_indiv');
    for imodel = 1:numel(models2get)
        Rpred.(models2get{imodel}).r2 = [Rpred.(models2get{imodel}).r2 Rpred_indiv.(models2get{imodel}).Rsquared];
    end
end

%%
figure(1); clf
mbase = 'stimsac';
mtest = 'stimrunsac';
r2base = max(Rpred.(mbase).r2, -.1);
r2test = max(Rpred.(mtest).r2, -.1);

plot(r2base,r2test, '.'); hold on
iix = max(r2base, r2test) < 0;
plot(r2base(iix), r2test(iix), '.', 'Color', repmat(.5, 1, 3))
plot(xlim, xlim, 'k')
xlabel(mbase)
ylabel(mtest)
xlim([-.1 .6])
ylim([-.1 .6])
title(subj)

% sum(Rpred.stimsac.r2 > 0)

%%
figure(1); clf; 
plot(Rpred_indiv.data.Robs, 'k')
hold on
plot(Rpred_indiv.data.gdrive, 'g')



