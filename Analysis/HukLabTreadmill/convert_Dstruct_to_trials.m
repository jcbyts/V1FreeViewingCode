function [Stim, eyePos] = convert_Dstruct_to_trials(D, varargin)
% [Stim, eyePos] = convert_Dstruct_to_trials(D, varargin)
% Inputs:
%   D <struct> the struct from export_gratings_for_py
% Optional Arguments:
%   bin_size (seconds): default = 1/60
%   pre_stim (seconds): time before grating onset (default = .2)
%   post_stim (seconds): time after grating offset (default = .2)
%
% Output:
%   Stim <struct> Struct of trial-aranged binned stimulus covariables


ip = inputParser();
ip.addParameter('bin_size', 1/60)
ip.addParameter('pre_stim', .2)
ip.addParameter('post_stim', .2)
ip.addParameter('plotit', false)
ip.addParameter('shuffle', false)
ip.parse(varargin{:})

bin_size = ip.Results.bin_size; % time resolution to bin stimulus and spikes
pre_stim = ip.Results.pre_stim;
post_stim = ip.Results.post_stim;
plotit = ip.Results.plotit;


warning off

circdiff = @(x,y) angle(exp(1i*(x - y)/180*pi))/pi*180;

grating_duration = mode(D.GratingOffsets-D.GratingOnsets);
fprintf('Grating Duration %02.2f s\n', grating_duration)
if isnan(grating_duration) || grating_duration < .1
    Stim = [];
    return
end

if isempty(D.frameTimes)
    Stim = [];
    return
end

num_trials = numel(D.GratingOnsets);
trial_time = -pre_stim:bin_size:(grating_duration + post_stim);
num_bins = numel(trial_time);

Stim = struct('bin_size', bin_size, ...
    'trial_time', trial_time, ...
    'grating_onsets', zeros(num_trials,1), ...
    'contrast', zeros(num_bins, num_trials), ...
    'phase_grating', zeros(num_bins, num_trials), ...
    'phase_eye', zeros(num_bins, num_trials), ...
    'freq', zeros(num_bins, num_trials), ...
    'direction', zeros(num_bins, num_trials), ...
    'speed_grating', zeros(num_bins, num_trials), ...
    'speed_eye', zeros(num_bins, num_trials), ...
    'tread_speed', zeros(num_bins, num_trials), ...
    'saccade_onset', zeros(num_bins, num_trials), ...
    'saccade_offset', zeros(num_bins, num_trials), ...
    'eye_pos', zeros(num_bins, num_trials, 2), ...
    'eye_pupil', zeros(num_bins, num_trials), ...
    'eye_pos_proj', zeros(num_bins, num_trials), ...
    'eye_pos_proj_adjusted', zeros(num_bins, num_trials));

eyeProj = cell(num_trials,1);

% itrial = itrial + 1;
% if itrial > num_trials
%     itrial = 1;
% end
if ip.Results.shuffle
    rng(1)
    trial_idx = randperm(num_trials);
else
    trial_idx = 1:num_trials;
end

trial_idx = trial_idx(~isnan(D.GratingOffsets(trial_idx)));

for itrial = 1:num_trials
%     fprintf('Trial %d/%d\n', itrial, num_trials)
    ttime = D.GratingOnsets(itrial) + trial_time;
    
    iix = D.frameTimes >= ttime(1) & D.frameTimes <= ttime(end);
    eix = D.eyeTime >= ttime(1) & D.eyeTime <= ttime(end);
    eyeTime = D.eyeTime(eix);
    eyePos = D.eyePos(eix,1:2);
    pupil = D.eyePos(eix,3);
    eyeLabels = D.eyeLabels(eix);
    
    eyeVel = [1000; abs(diff(imgaussfilt(hypot(eyePos(:,1), eyePos(:,2)), 10)))*1000];
    
    eyeLabels(imboxfilt(double(eyeVel > 5), 5)>0) = 2;
    
    frameTimes = D.frameTimes(iix);
    framePhase = D.framePhase(iix);
    
    dPhase = circdiff(framePhase(2:end) , framePhase(1:end-1));
    
    % GRATING DIRECTION - can be shuffled -- control fo all eye projection
    % measures 
    gratingDirection = D.GratingDirections(trial_idx(itrial));
    
    gratingContrast = interp1(frameTimes, D.frameContrast(iix), ttime, 'nearest');
    gratingdPhase = interp1(frameTimes(2:end), dPhase, ttime, 'next');
    gratingdPhase(trial_time <=0) = 0;
    
    etime = eyeTime - D.GratingOnsets(itrial);
    
    if plotit
        figure(1); clf %#ok<*UNRCH>
        subplot(3,1,1)
        plot(trial_time, gratingPhase, '-o')
        plot(trial_time, gratingdPhase, '-o')
        legend('\Delta phase')
        hold on
        
        subplot(3,1,2)
        plot(trial_time, gratingContrast, '-o')
        legend('contrast')
        
        subplot(3,1,3)
        plot(etime, eyePos)
        hold on
        plot(etime(eyeLabels==1), eyePos(eyeLabels==1,:), '.')
        xlim(trial_time([1 end]))
        xlabel('Time from Grating Onset')
    end
    
    u = [cosd(gratingDirection) sind(gratingDirection)]; % grating direction vector
    
    if plotit
        figure(2); clf
        quiver(0, 0, u(1)*2, u(2)*2, 'Linewidth', 2); hold on
        xlim([-10 10])
        ylim([-10, 10])
        plot(eyePos(eyeLabels==1,1), eyePos(eyeLabels==1,2), '.k')
        ix = eyeLabels==1 & etime > 0 & etime < grating_duration;
        plot(eyePos(ix, 1), eyePos(ix,2), '.r')
        title(gratingDirection)
    end
    
    % compute eye position projected on grating direction
    eye_proj = eyePos*u';
    
    fixations = bwlabel(eyeLabels==1);
    lastfix = 0;
    
    % saccade-adjusted smoothed eye position
    eye_proj_smooth = eye_proj.*double(eyeLabels==1);
    
    for ifix = 1:max(fixations)
        fixix = fixations==ifix;
        tmp = imgaussfilt(eye_proj_smooth(fixix), 15);
        eye_proj(fixix) = tmp;
        
        tmp = tmp - tmp(1) + lastfix;
        eye_proj_smooth(fixix) = tmp;
        
        
        lastfix = tmp(end);
    end
    
    eye_proj_smooth(eyeLabels~=1) = nan;
    % convert eye position to phase
    cycle_width = 1/D.GratingFrequency(itrial);
    eye_proj_phase = mod(-eye_proj, cycle_width)/cycle_width*360;
    
    % smooth eye phase
    epiwrap = eye_proj_phase/180*pi-pi;
    epiun = unwrap(epiwrap);
    eye_phase_smooth = mod(imgaussfilt(epiun, 10)/pi*180 + 180, 360);
    
    
    if plotit
        figure(10); clf
        subplot(3,1,1)
        plot(etime, eye_proj)
        xlabel('Time from grating onset')
        ylabel('Degrees')
        subplot(3,1,2)
        plot(etime, eye_proj_smooth)
        xlabel('Time from grating onset')
        ylabel('Degrees (saccade-adjusted)')
        subplot(3,1,3)
        plot(etime, eye_proj_phase)
        hold on
        plot(etime, eye_phase_smooth)
        xlabel('Time from grating onset')
        ylabel('Eye Phase (of Grating)')
    end
    
    eyeProj{itrial} = eye_proj_smooth;
    
    eye_speed = [0; (eye_proj_smooth(2:end) - eye_proj_smooth(1:end-1))*1e3];
    
    gphase = interp1(frameTimes, mod(framePhase, 360), ttime, 'nearest');
    ephase = interp1(eyeTime, eye_phase_smooth, ttime, 'nearest');
    
    dephase = [0; circdiff(eye_phase_smooth(2:end), eye_phase_smooth(1:end-1))];
    
    eye_speed = nanmean(toeplitz(eye_speed, eye(15)),2);
    
    total_phase = gphase + ephase;
    dphase_total = [0 circdiff(total_phase(2:end) , total_phase(1:end-1))];
    dphase_grat = [0 circdiff(gphase(2:end) , gphase(1:end-1))];
    
    stim_on = double(gratingContrast>0);
    stim_on = [0 stim_on(1:end-1)];
    
    sac_binned = interp1(etime, double(eyeLabels==1), trial_time, 'nearest');
    
    sac_onsets = [0 diff(sac_binned)== -1];
    sac_offsets = [0 diff(sac_binned)==1];
    
    % treadmill
    trix = D.treadTime >= ttime(1)-.1 & D.treadTime <= ttime(end)+.1;
    treadtime = D.treadTime(trix);
    treadSpeed = D.treadSpeed(trix);
    if numel(treadtime) >= .5*num_bins
        tread_speed = interp1(treadtime, treadSpeed, ttime);
    else
        tread_speed = nan;
    end
    
    if plotit
        figure(11); clf
        subplot(3,1,1)
        plot(ttime, gphase.*stim_on, '-'); hold on
        plot(ttime, mod(ephase, 360))
        plot(ttime, mod(gphase + ephase, 360))
        
        subplot(3,1,2)
        plot(ttime, dphase_total.*stim_on); hold on
        plot(ttime, dphase_grat.*stim_on)
        
        subplot(3,1,3)
        plot(trial_time, D.GratingSpeeds(itrial)*stim_on); hold on
        plot(trial_time, -dphase_grat/360/D.GratingFrequency(itrial)/bin_size)
        plot(trial_time, -dphase_total/360/D.GratingFrequency(itrial)/bin_size)
    end
    
    
    % store variables
    Stim.grating_onsets(itrial) = D.GratingOnsets(itrial);
    Stim.contrast(:,itrial) = gratingContrast;
    Stim.direction(:,itrial) = gratingDirection*stim_on;
    Stim.direction(~stim_on,itrial) = nan;
    
    Stim.freq(:,itrial) = D.GratingFrequency(itrial)*stim_on;
    Stim.speed_grating(:,itrial) = D.GratingSpeeds(itrial)*stim_on;
    Stim.speed_eye(:,itrial) = interp1(etime, eye_speed, trial_time);
    Stim.phase_grating(:,itrial) = gphase;
    Stim.phase_eye(:,itrial) = ephase;
    Stim.eye_pos(:,itrial,:) = interp1(etime, eyePos, trial_time);
    Stim.eye_pupil(:,itrial) = interp1(etime, pupil, trial_time);
    Stim.eye_pos_proj(:,itrial) = interp1(etime, eye_proj, trial_time, 'nearest');
    Stim.eye_pos_proj_adjusted(:,itrial) = interp1(etime, eye_proj_smooth, trial_time, 'nearest');
    Stim.saccade_onset(:,itrial) = sac_onsets;
    Stim.saccade_offset(:,itrial) = sac_offsets;
    Stim.tread_speed(:,itrial) = tread_speed;
    
    
end

warning on