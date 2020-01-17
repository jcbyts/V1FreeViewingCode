function [stim, spks, opts, params_stim, kx, ky] = preprocess_grating_subspace_data(Exp, varargin)
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
ip.addParameter('stim_field', 'ITI')
ip.addParameter('trial_inds', [])
ip.parse(varargin{:})

stimField = 'Grating';

opts = ip.Results;
opts.fs_spikes = opts.fs_stim;
opts.up_samp_fac = 1;

validTrials = io.getValidTrials(Exp, stimField);


ori = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,2), Exp.D(validTrials), 'uni', 0));
cpd = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,3), Exp.D(validTrials), 'uni', 0));
frameTime = Exp.ptb2Ephys(cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,1), Exp.D(validTrials), 'uni', 0)));
% eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
NT = numel(frameTime);
% eyeFrame = nan(NT,1);
% eyeFrame(1) = find(eyeTime > frameTime(1), 1);
% for iFrame = 2:NT
%     eyeFrame(iFrame) = eyeFrame(iFrame-1) + find(eyeTime(eyeFrame(iFrame-1):end) > frameTime(iFrame),1 , 'first');
% end

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

% [kx,ky] = pol2cart(ori/180*pi, cpd);
kx = ori;
ky = cpd;

opts.oris = oris;
opts.cpds = cpds;
opts.dim =[nori ncpd];
num_kx = nori;
num_ky = ncpd;
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
    spks = downsample_time(spks, t_downsample);
end


% Track stimulus parameters in a format that is usable by NIM class from
% Dan Butts' group
stim_dx = 1; %default unitless spatial resolution
tent_spacing = []; %default no tent-bases
boundary_conds = [0 0 0]; %tied to 0 in all dims
split_pts = []; %no internal boundaries

params_stim = struct('dims',[opts.num_lags_stim, num_ky, num_kx],...
    'dt',1e3/opts.fs_stim,'dx',stim_dx,'up_fac',opts.up_samp_fac,...
    'tent_spacing',tent_spacing,'boundary_conds',boundary_conds,...
    'split_pts',split_pts);
        
params_stim(2) = struct('dims',[opts.num_lags_sac_pre, 1],...
    'dt',1e3/opts.fs_spikes,'dx',stim_dx,'up_fac',opts.up_samp_fac,...
	'tent_spacing',tent_spacing,'boundary_conds',boundary_conds,...
    'split_pts',split_pts);

params_stim(3) = struct('dims',[opts.num_lags_sac_post, 1],...
    'dt',1e3/opts.fs_spikes,'dx',stim_dx,'up_fac',opts.up_samp_fac,...
	'tent_spacing',tent_spacing,'boundary_conds',boundary_conds,...
    'split_pts',split_pts);
        
params_stim(4) = struct('dims',[opts.num_lags_sac_post, 1],...
    'dt',1e3/opts.fs_spikes,'dx',stim_dx,'up_fac',opts.up_samp_fac,...
	'tent_spacing',tent_spacing,'boundary_conds',boundary_conds,...
    'split_pts',split_pts);

opts.num_kx = num_kx;
opts.num_ky = num_ky;
opts.kxs = oris;
opts.kys = cpds;