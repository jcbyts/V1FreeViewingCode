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