function do_regression_ss(Stim, D, cid, fout)

fname = fullfile(fout, sprintf('%s_%d.mat', D.subj, cid));

if exist(fname, 'file')
    return
end

try
    iix = D.spikeIds == cid;

    st = D.spikeTimes(iix);

    trialidx = find(Stim.grating_onsets > min(st) & Stim.grating_onsets < max(st));


    sp = struct('spikeTimes', st, 'spikeIds', ones(size(st)));
    Robs = bin_spikes_at_frames(Stim, sp, 0, trialidx);

    if (mean(Robs) / Stim.bin_size) < 1
        return
    end

    NC = size(Robs,2);
    U = eye(NC);
    cc = 0;

    opts = struct();
    folds = 5;
    opts.use_spikes_for_dfs = false;
    opts.folds = folds;
    opts.trialidx = trialidx;
    opts.spike_smooth = 5;

    Robs = filtfilt(ones(opts.spike_smooth,1)/opts.spike_smooth, 1, Robs);
    %% get data filters
    if opts.use_spikes_for_dfs

        Rt = reshape(Robs, numel(Stim.trial_time), [], NC);
        figure(1); clf;
        plot(squeeze(sum(sum(Rt,2),1)), sum(Robs), '.'); hold on; plot(xlim, xlim, 'k')

        Rt = squeeze(sum(Rt));

        dfs = false(numel(Stim.trial_time), numel(Stim.grating_onsets), NC);
        dfsTrials = false(numel(Stim.grating_onsets), NC);
        for cc = 1:NC
            iix = getStableRange(imboxfilt(Rt(:,cc), 25));
            dfsTrials(iix,cc) = true;
            dfs(:,iix,cc) = true;
            drawnow
        end

        dfs = reshape(dfs, [], NC);
    end

    %% build the design matrix components
    % build design matrix
    assert(Stim.bin_size==1/60, 'Parameters have only been set for 60Hz sampling')
    opts.collapse_speed = true;
    opts.collapse_phase = true;
    opts.include_onset = false;
    opts.stim_dur = median(D.GratingOffsets-D.GratingOnsets) + 0.2; % length of stimulus kernel
    opts.use_sf_tents = false;

    stim_dur = ceil((opts.stim_dur)/Stim.bin_size);
    opts.stim_ctrs = [2:5:10 15:15:stim_dur];
    if ~isfield(opts, 'stim_ctrs')
        opts.stim_ctrs = [0:5:10 15:10:stim_dur-2];
    end
    Bt = tent_basis(0:stim_dur+15, opts.stim_ctrs);

    [X, opts] = build_design_matrix(Stim, opts);

    % concatenate full design matrix
    label = 'Stim';
    regLabels = {label};
    k = 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = repmat(k, 1, size(X_,2)); %#ok<REPMAT>
    fullR = X_;

    if opts.include_onset
        label = 'Stim Onset';
        regLabels = [regLabels {label}];
        k = k + 1;
        X_ = X{ismember(opts.Labels, label)};
        regIdx = [regIdx repmat(k, [1, size(X_,2)])];
        fullR = [fullR X_];
    end

    % add additional covariates
    label = 'Drift';
    regLabels = [regLabels {label}];
    k = k + 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = [regIdx repmat(k, [1, size(X_,2)])];
    fullR = [fullR X_];

    label = 'Saccade';
    regLabels = [regLabels {label}];
    k = k + 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = [regIdx repmat(k, [1, size(X_,2)])];
    fullR = [fullR X_];

    label = 'Running';
    regLabels = [regLabels {label}];
    k = k + 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = [regIdx repmat(k, [1, size(X_,2)])];
    fullR = [fullR X_];

    isrunning = reshape(Stim.tread_speed(:, opts.trialidx), [], 1) > opts.run_thresh;
    isrunning = sign(isrunning - .5);

    zpupil = reshape(Stim.eye_pupil(:, opts.trialidx), [], 1) / nanstd(reshape(Stim.eye_pupil(:, opts.trialidx), [], 1)); %#ok<*NANSTD>
    zpupil = zpupil - nanmean(zpupil); %#ok<*NANMEAN>
    ispupil = zpupil > opts.pupil_thresh;
    ispupil = sign(ispupil - .5);

    regLabels = [regLabels {'Is Running'}];
    k = k + 1;
    regIdx = [regIdx k];
    fullR = [fullR isrunning(:)];

    regLabels = [regLabels {'Is Pupil'}];
    k = k + 1;
    regIdx = [regIdx k];
    fullR = [fullR ispupil(:)];

    assert(size(fullR,2) == numel(regIdx), 'Number of covariates does not match Idx')
    assert(numel(unique(regIdx)) == numel(regLabels), 'Number of labels does not match')

    for i = 1:numel(regLabels)
        label= regLabels{i};
        cov_idx = regIdx == find(strcmp(regLabels, label));
        fprintf('[%s] has %d parameters\n', label, sum(cov_idx))
    end

    %% Find valid time range using stim gain model

    Rpred_indiv = struct();
    Rpred_indiv.data = struct();

    % try different models
    modelNames = {'nostim', 'stim', 'stimsac', 'stimrun', 'stimrunsac', 'drift'};

    excludeParams = { {'Stim', 'Stim Onset'}, {'Running', 'Saccade'}, {'Running'}, {'Saccade'}, {}, {'Stim','Stim Onset','Running','Saccade'} };
    alwaysExclude = {'Stim R', 'Stim S', 'Is Running', 'Is Pupil'};

    % models2fit = {'drift', 'stim'};
    models2fit = modelNames;

    Ntotal = size(Robs,1);
    Rpred_indiv.data.indices = false(Ntotal, NC);
    Rpred_indiv.data.Robs = Robs;
    Rpred_indiv.data.gdrive = nan(Ntotal,NC);
    Rpred_indiv.data.Rpred = nan(Ntotal,NC)';
    Rpred_indiv.data.Gain = nan(2,NC);
    Rpred_indiv.data.Ridge = nan(1,NC);

    for iModel = find(ismember(modelNames, models2fit))
        Rpred_indiv.(modelNames{iModel}).Rpred = nan(Ntotal,NC)';
        Rpred_indiv.(modelNames{iModel}).Offset = nan(folds,NC);
        Rpred_indiv.(modelNames{iModel}).Beta = cell(folds,1);
        Rpred_indiv.(modelNames{iModel}).Ridge = nan(1, NC);
        Rpred_indiv.(modelNames{iModel}).Rsquared = nan(1,NC);
        Rpred_indiv.(modelNames{iModel}).CC = nan(1,NC);
    end

    % Fit Gain models for running and pupil
    GrestLabels = [{'Stim Onset'}    {'Drift'}    {'Saccade'}    {'Running'}];
    GainModelNames = {'RunningGain', 'PupilGain'}; %{'nostim', 'stim', 'stimsac', 'stimrun', 'stimrunsac'};
    GainTerm = {'Is Running', 'Is Pupil'};

    for iModel = 1:numel(GainTerm)

        labelIdx = ismember(regLabels, [{'Stim'} GrestLabels]);
        covIdx = regIdx(ismember(regIdx, find(labelIdx)));
        covLabels = regLabels(labelIdx);
        [~, ~, covIdx] = unique(covIdx);

        Rpred_indiv.(GainModelNames{iModel}).covIdx = covIdx;
        Rpred_indiv.(GainModelNames{iModel}).Beta = cell(folds,1);
        Rpred_indiv.(GainModelNames{iModel}).Offset = zeros(folds, NC);
        Rpred_indiv.(GainModelNames{iModel}).Gains = zeros(folds, 2, NC);
        Rpred_indiv.(GainModelNames{iModel}).Labels = covLabels;
        Rpred_indiv.(GainModelNames{iModel}).Rpred = zeros(NC, Ntotal);
        Rpred_indiv.(GainModelNames{iModel}).Ridge = zeros(folds, NC);
        Rpred_indiv.(GainModelNames{iModel}).Rsquared = nan(1,NC);
        Rpred_indiv.(GainModelNames{iModel}).CC = nan(1,NC);

        ndim = numel(covIdx);
        for ifold = 1:folds
            Rpred_indiv.(GainModelNames{iModel}).Beta{ifold} = zeros(ndim, NC);
        end
    end

    for cc = 1:NC

        tread_speed = reshape(Stim.tread_speed(:,opts.trialidx), [], 1);
        good_inds = find(~isnan(tread_speed));
        tread_speed = tread_speed(good_inds);

        restLabels = [{'Stim Onset'}    {'Drift'}];
        Lgain = nan;
        Lfull = nan;
        X = fullR(good_inds,:);
        [Betas, Gain, Ridge, Rhat, ~, ~, gdrive] = AltLeastSqGainModel(X, Robs(good_inds,cc), 1:size(X,1), ...
            regIdx, regLabels, {'Stim'}, 'Drift', restLabels, Lgain, Lfull);


        figure(3); clf
        plot(Robs(good_inds,cc), 'k'); hold on
        plot(Rhat, 'r', 'Linewidth', 1)
        plot(gdrive./max(gdrive)*max(Rhat), 'g', 'Linewidth', 2)
        plot(gdrive > prctile(gdrive, 75)/2, 'b', 'Linewidth', 2)
        plot(tread_speed ./ max(tread_speed) * max(Rhat), 'c')
        df = gdrive > prctile(gdrive, 75)/2;

        Rpred_indiv.data.gdrive(good_inds,cc) = gdrive;
        Rpred_indiv.data.Rpred(cc,good_inds) = Rhat;
        Rpred_indiv.data.Ridge(cc) = Ridge;
        Rpred_indiv.data.Gain(:,cc) = Gain;

        % Build cv indices that are trial-based
        good_inds = intersect(good_inds, find(df));

        Rpred_indiv.data.indices(good_inds,cc) = true;

        good_trials = (1:numel(Stim.grating_onsets))';  %find(dfsTrials(:,cc));
        num_trials = numel(good_trials);

        folds = 5;
        n = numel(Stim.trial_time);
        T = size(Robs,1);

        dataIdxs = true(folds, T);
        rng(1)

        trorder = randperm(num_trials);

        for t = 1:(num_trials)
            i = mod(t, folds);
            if i == 0, i = folds; end
            dataIdxs(i,(good_trials(trorder(t))-1)*n + (1:n)) = false;
        end

        % good_inds = intersect(good_inds, find(sum(dataIdxs)==(folds-1)));
        fprintf('%d good samples\n', numel(good_inds))
        plot(good_inds, ones(numel(good_inds),1), '.')


        R = Robs(good_inds,cc);
        X = fullR(good_inds,:);

        figure(1); clf
        plot(Robs(:,cc), 'k'); hold on

        for iModel = find(ismember(modelNames, models2fit))

            fprintf('Fitting Model [%s]\n', modelNames{iModel})

            exclude = [excludeParams{iModel} alwaysExclude];
            modelLabels = setdiff(regLabels, exclude); %#ok<*NASGU>
            %     evalc("[Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR(good_inds,:), Robs(good_inds,:)', modelLabels, regIdx, regLabels, folds, dataIdxs);");
            % dataIdxs(:,df)

            [Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(X, R', modelLabels, regIdx, regLabels, folds, dataIdxs(:,good_inds));

            Rpred_indiv.(modelNames{iModel}).covIdx = fullIdx;
            Rpred_indiv.(modelNames{iModel}).Labels = fullLabels;
            Rpred_indiv.(modelNames{iModel}).Rpred(cc,good_inds) = Vfull;
            Rpred_indiv.(modelNames{iModel}).Offset(:,cc) = cellfun(@(x) x(1), fullBeta(:));
            for i = 1:numel(fullBeta)
                fullBeta{i}(1) = [];
                Rpred_indiv.(modelNames{iModel}).Beta{i}(:,cc) = fullBeta{i}; %#ok<*NODEF>
            end

            Rpred_indiv.(modelNames{iModel}).Ridge(cc) = fullRidge;
            Rpred_indiv.(modelNames{iModel}).Rsquared(cc) = rsquared(R, Vfull'); %compute explained variance
            Rpred_indiv.(modelNames{iModel}).CC(cc) = corr(R, Vfull'); %modelCorr(R,Vfull,1); %compute explained variance
        end



        Lgain = nan;
        Lfull = nan;

        for iModel = 1:numel(GainTerm)
            for ifold = 1:folds
                train_inds = find(dataIdxs(ifold,good_inds))';
                test_inds = find(~dataIdxs(ifold,good_inds))';

                %             evalc("[Betas, Gain, Ridge, Rhat, Lgain, Lfull] = AltLeastSqGainModel(fullR(good_inds,:), Robs(good_inds,cc), train_inds, regIdx, regLabels, {'Stim'}, GainTerm(iModel), restLabels, Lgain, Lfull);");
                [Betas, Gain, Ridge, Rhat, Lgain, Lfull] = AltLeastSqGainModel(X, R, train_inds, regIdx, regLabels, {'Stim'}, GainTerm(iModel), GrestLabels, Lgain, Lfull);

                Rpred_indiv.(GainModelNames{iModel}).Offset(ifold,cc) = Betas(1);
                Rpred_indiv.(GainModelNames{iModel}).Beta{ifold}(:,cc) = Betas(2:end);
                Rpred_indiv.(GainModelNames{iModel}).Gains(ifold,:,cc) = Gain;
                Rpred_indiv.(GainModelNames{iModel}).Ridge(ifold,cc) = Ridge;
                Rpred_indiv.(GainModelNames{iModel}).Rpred(cc,good_inds(test_inds)) = Rhat(test_inds);
            end
            % evaluate model
            Rpred_indiv.(GainModelNames{iModel}).Rsquared(cc) = rsquared(R, Rpred_indiv.(GainModelNames{iModel}).Rpred(cc,good_inds)'); %compute explained variance
            Rpred_indiv.(GainModelNames{iModel}).CC(cc) = corr(R, Rpred_indiv.(GainModelNames{iModel}).Rpred(cc,good_inds)'); %compute explained variance
            fprintf('Done\n')
        end

        %     drawnow
    end


    save(fname, '-v7.3', 'Rpred_indiv', 'opts')

end