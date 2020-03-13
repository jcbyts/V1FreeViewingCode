function [stim, spks, opts] = preprocess_grating_subspace_data(Exp, varargin)
% Preprocess data for analysis
% Bin the stimulus / spikes and keep track of all parameters used
% Inputs:
%   D (data struct) - has fields corresponding to the stimulus of interest
%                     and spikes
%   Optional Arguments:
%       'fs_stim'           - sampling resolution to bin the stimulus at
%       'up_samp_fac',      - multiplier to bin spikes at higher resolution
%       'num_lags_stim'     - number of lags to include for a stimulus filter
%       'num_lags_sac_post' - number of lags after a saccade (in spike
%                             sampling resolution)
%       'num_lags_sac_pre'  - number of lags before a saccade (in spike
%                             resolution)
%       'use_mirror'        - only analyze half the hartley basis set
%                             (default: true). This cuts the number of
%                             parameters in half, but is only valid if the
%                             gratings are not drifting
%       'true_zero'         - The origin of the fourier domain is currently
%                             one of the hartley stimulus options. Zero it
%                             out because it is really just a gray screen.
%       'trial_inds'        - list of trials to analyze
% Outputs:
%   stim [cell array]   - (T x m) stimuli: stimulus, saccade_onsets, saccade_offsets, eye velocity
%   spks,[T x n]        - binned spike counts
%   opts [struct]       - information about the preprocessing
%   params [struct]     - information about the stim fields
%   kx, ky, frame_binned , [T x 1] the stimulus and corresponding bins
%
% Example:
%   [stim, spks, opts, params_stim] = io.preprocess_grating_subspace_data(D, 'stim_field', 'hartleyFF', 'fs_stim', 120);

% jly 2020
ip = inputParser();
ip.KeepUnmatched = true;
ip.addParameter('fs_stim', 120)      % stimulus binning rate (per second)
ip.addParameter('up_samp_fac', 1)    % spikes are sampled at higher temporal res?
ip.addParameter('num_lags_stim', 20) % number of lags for stimulus filter
ip.addParameter('num_lags_sac_post', 50)  % number of time lags to capture saccade effect
ip.addParameter('num_lags_sac_pre', 50)  % number of time lags to capture saccade effect
ip.addParameter('stim_field', 'Grating')
ip.addParameter('trial_inds', [])
ip.parse(varargin{:})

stimField = ip.Results.stim_field;

opts = ip.Results;
opts.fs_spikes = opts.fs_stim;
opts.up_samp_fac = 1;

validTrials = io.getValidTrials(Exp, stimField);

frameTime = cellfun(@(x) Exp.ptb2Ephys(x.PR.NoiseHistory(:,1)), Exp.D(validTrials), 'uni', 0);
ori = cellfun(@(x) x.PR.NoiseHistory(:,2), Exp.D(validTrials), 'uni', 0);
cpd = cellfun(@(x) x.PR.NoiseHistory(:,3), Exp.D(validTrials), 'uni', 0);

trialDuration = cellfun(@numel , frameTimes);
trialStartFrameNum = 1 + cumsum([0; trialDuration(1:end-1)]);
trialEndFrameNum = trialDuration + trialStartFrameNum - 1;

% frozenSequence starts, duration
frozenTrials = cellfun(@(x) x.PR.frozenSequence, Exp.D(trialInds));
if ~any(frozenTrials)
    frozen_seq_starts = [];
    frozen_seq_dur = 0;
else
    frozen_seq_dur = arrayfun(@(x) x.frozenSequenceLength, trial(trialInds(frozenTrials)));
    frozen_seq_dur = min(frozen_seq_dur);
    frozen_seq_starts = [];
    for iTrial = find(frozenTrials(:))'
        numRepeats = floor(trialDuration(iTrial)/frozen_seq_dur);
        trial_seq_starts = trialStartFrameNum(iTrial) + (0:(numRepeats-1))*frozen_seq_dur;
        frozen_seq_starts = [frozen_seq_starts trial_seq_starts]; %#ok<AGROW>
    end
end

frameTime = cell2mat(frameTime);
ori = cell2mat(ori);
cpd = cell2mat(cpd);

NT = numel(frameTime);

% --- concatenate different sessions
ts = frameTimes(trialStartFrameNum);
te = frameTimes(trialEndFrameNum);

ix = ~isnan(cpd);
cpds = unique(cpd(ix));
oris = unique(ori(ix));
ncpd = numel(cpds);
nori = numel(oris);

[~, oriid] = max(ori==oris', [], 2);
[~, cpdid] = max(cpd==cpds', [], 2);

blank = cpdid==1;
ncpd = ncpd - 1;
cpdid = cpdid - 1;
cpds(1) = [];

% find discontinuous fragments in the stimulus (trials)
fn_stim   = ceil((te-ts)*opts.fs_stim);

% convert times to samples
frame_binned = io.convertTimeToSamples(frameTimes, opts.fs_stim, ts, fn_stim);

% find frozen repeat indices
opts.frozen_repeats = frame_binned(bsxfun(@plus, frozen_seq_starts(:), 0:frozen_seq_dur));

% --- cleanup frozen trials (remove any that are wrong)
% [sequence, ~, ic] = unique(kx(opts.frozen_repeats), 'rows', 'sorted');
% opts.frozen_repeats(ic~=1,:) = []; 
C = nancov([ori(opts.frozen_repeats) cpd(opts.frozen_repeats)]');
margV = diag(C); C = C ./ sqrt((margV(:)*margV(:)'));
[~, ref] = max(sum(C > .99, 2));
keep = C(ref,:) > .99;
opts.frozen_repeats = opts.frozen_repeats(keep,:);


opts.oris = oris;
opts.cpds = cpds;
opts.dim =[nori ncpd];

ind = sub2ind([nori ncpd], oriid(~blank), cpdid(~blank));

binsize = 1./Exp.S.frameRate;


ft = (1:NT)';
stim{1} = full(sparse(ft(~blank), ind, ones(numel(ind),1), NT, nori*ncpd));
spks = binNeuronSpikeTimesFast(Exp.osp, frameTime, binsize);

% saccade times
sacoffset = ip.Results.num_lags_sac_pre/ip.Results.fs_stim;
stim{2} = full(binTimesFast(Exp.slist(:,1)-sacoffset, frameTime, binsize));
stim{3} = full(binTimesFast(Exp.slist(:,2), frameTime, binsize));


t_downsample = Exp.S.frameRate / ip.Results.fs_stim;
if t_downsample > 1
    stim{1} = downsample_time(stim{1}, t_downsample) / t_downsample;
    stim{2} = downsample_time(stim{2}, t_downsample) / t_downsample;
    stim{3} = downsample_time(stim{3}, t_downsample) / t_downsample;
    frameTime = downsample(frameTime, t_downsample);
    ori = downsample(ori, t_downsample);
    cpd = downsample(cpd, t_downsample);
    spks = downsample_time(spks, t_downsample);
    frameTime = frameTime(1:size(spks,1));
    ori = ori(1:size(spks,1));
    cpd = cpd(1:size(spks,1));
end

opts.ori = ori;
opts.cpd = cpd;
opts.frameTime = frameTime;