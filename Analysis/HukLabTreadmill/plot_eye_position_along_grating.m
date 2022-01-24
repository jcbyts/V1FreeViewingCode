function [S, fig] = plot_eye_position_along_grating(D, bin_size)
% plots the eye position projected on the grating direction with saccades
% removed
% [S, fig] = plot_eye_position_along_grating(D, bin_size)

if ~exist('bin_size', 'var')
    bin_size = 1/60;
end

%% Get all relevant covariates
Stim = convert_Dstruct_to_trials(D, 'bin_size', bin_size, 'pre_stim', .2, 'post_stim', .2);
StimShuff = convert_Dstruct_to_trials(D, 'bin_size', bin_size, 'pre_stim', .2, 'post_stim', .2, 'shuffle', true);

%% plot
fig = figure();
set(gcf, 'Color', 'w')

Stim.eye_pos_proj_adjusted(abs(Stim.eye_pos_proj_adjusted) > 1) = nan; %remove leftover saccades
StimShuff.eye_pos_proj_adjusted(abs(StimShuff.eye_pos_proj_adjusted) > 1) = nan; %remove leftover saccades

ttime = Stim.trial_time;
cmap = lines;

speeds = unique(max(Stim.speed_grating));
nspeeds = numel(speeds);

label = {};
S = struct();

clear h
for i = 1:nspeeds
    iix = max(Stim.speed_grating)==speeds(i);
    m = nanmean(Stim.eye_pos_proj_adjusted(:, iix),2);
    sd = nanstd(Stim.eye_pos_proj_adjusted(:, iix), [], 2)/sqrt(sum(iix));
    
    field = sprintf('speed%d', speeds(i));
    S.(field).time = ttime;
    S.(field).mean = m;
    S.(field).stderr = sd;
    
    m = interp1(find(~isnan(m)), m(~isnan(m)), 1:numel(m));
    sd = interp1(find(~isnan(sd)), sd(~isnan(sd)), 1:numel(sd));
    
    h(i) = plot.errorbarFill(ttime(~isnan(m)), m(~isnan(m)), sd(~isnan(m)), 'k', 'EdgeColor', 'none', 'FaceAlpha', .5, 'FaceColor', cmap(i,:)); hold on
    plot(ttime, m, 'Color', cmap(i,:));
    label{i} = sprintf('Grating Speed = %d (deg/s)', speeds(i));
    
end

if exist('StimShuff', 'var')
    iix = max(Stim.speed_grating)>0;
    m = nanmean(StimShuff.eye_pos_proj_adjusted(:, iix),2);
    sd = nanstd(StimShuff.eye_pos_proj_adjusted(:, iix), [], 2)/sqrt(sum(iix));
    m = interp1(find(~isnan(m)), m(~isnan(m)), 1:numel(m));
    sd = interp1(find(~isnan(sd)), sd(~isnan(sd)), 1:numel(sd));
    
    S.shuffle.time = ttime;
    S.shuffle.mean = m;
    S.shuffle.stderr = sd;

    h(numel(h)+1) = plot.errorbarFill(ttime(~isnan(m)), m(~isnan(m)), sd(~isnan(m)), 'k', 'EdgeColor', 'none', 'FaceAlpha', .5, 'FaceColor', repmat(.5, 1, 3)); hold on
    label{i+1} = 'Shuffled';
    plot(ttime, m, 'Color', repmat(.5, 1, 3));
end

legend(h, label, 'Location', 'Best')
xlabel('Time from Grating Onset (s)')
ylabel('Eye position projected on Grating direction (deg)')