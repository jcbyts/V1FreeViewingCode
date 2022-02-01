function [X, opts] = build_design_matrix(Stim, opts)
% [X, opts] = build_design_matrix(Stim, opts)

X = [];
Labels = [];

defopts = struct();

defopts.stim_dur = 1; 
defopts.spd_ctrs = [1 2];
defopts.sf_ctrs = [1 3];
defopts.use_onset_only = false;
defopts.use_derivative = true;
defopts.use_sf_tents = false;
defopts.include_onset = true;
defopts.include_full_stim_split = false;
defopts.dph = 45; % spacing for phase basis
defopts.collapse_speed = true;
defopts.collapse_phase = true;
defopts.pupil_thresh = 1;

defopts.phase_ctrs = 0:defopts.dph:360-defopts.dph;

% running parameters
defopts.run_thresh = 3;
defopts.nrunbasis = 25;
defopts.run_offset = -5;
defopts.run_post = 40;

% saccade parameters
defopts.nsacbasis = 20;
defopts.sac_offset = -10;
defopts.sac_post = 40;
defopts.sac_covariate = 'onset';

defopts.num_drift_tents = 15;

% merge requested options with the defaults
opts = mergeStruct(defopts,opts);

if isfield(opts, 'trialidx')
    numtrials = numel(Stim.grating_onsets);
    fields = fieldnames(Stim);
    for f = 1:numel(fields)
        sz = size(Stim.(fields{f}));
        tdim = find(sz==numtrials);
        if isempty(tdim)
            continue
        end

        switch tdim

            case 1
                Stim.(fields{f}) = Stim.(fields{f})(opts.trialidx);
            case 2
                if numel(sz)==3
                    Stim.(fields{f}) = Stim.(fields{f})(:,opts.trialidx,:);
                else
                    Stim.(fields{f}) = Stim.(fields{f})(:,opts.trialidx);
                end
        end
    end
end


opts.nphi = numel(opts.phase_ctrs);
opts.nspd = numel(opts.spd_ctrs);

if opts.collapse_speed
    opts.nspd = 1; % collapse across speed
end
if opts.collapse_phase
    opts.nphi = 1; % collapse across phase
end

circdiff = @(x,y) angle(exp(1i*(x - y)/180*pi))/pi*180;

contrast = Stim.contrast(:);
freq = Stim.freq(:);
NT = numel(freq);

stim_on = contrast > 0 & freq > 0;

% onset only
if opts.use_onset_only
%     stim_on = max(filter([1;-1], 1, stim_on), 0);
    stim_on = [false; diff(stim_on)>0];
end

direction = Stim.direction(:);
speed = Stim.speed_grating(:) .* stim_on;
speedeye = speed + Stim.speed_eye(:) .* stim_on;

opts.directions = unique(direction(stim_on));

opts.nd = numel(opts.directions);

% turn direction into "one hot" matrix
xdir = (direction == opts.directions(:)') .* stim_on;

if opts.use_sf_tents
    % spatial frequency (with basis)
    sf = tent_basis(freq, opts.sf_ctrs) .* stim_on;
    opts.nsf = numel(opts.sf_ctrs);
else
    % spatial frequency (no tents)
    opts.sf_ctrs = unique(freq(stim_on));
    opts.nsf = numel(opts.sf_ctrs);
    sf = (freq == opts.sf_ctrs(:)') .* stim_on;
end

spd = tent_basis(speedeye, opts.spd_ctrs) .* stim_on;

% figure(1); clf; plot(max(1- abs(circdiff((0:360)',phase_ctrs)/dph), 0))

xphase = max( 1 - abs(circdiff(Stim.phase_eye(:) + Stim.phase_grating(:), opts.phase_ctrs)/opts.dph), 0);

Xbig = zeros(NT, opts.nd, opts.nsf, opts.nphi, opts.nspd);
for idir = 1:opts.nd
    for isf = 1:opts.nsf
        if opts.nphi > 1
            for iphi = 1:opts.nphi
                if opts.nspd > 1
                    Xbig(:,idir, isf, iphi, :) = xdir(:,idir).*sf(:,isf).*xphase(:,iphi).*spd;
                else
                    Xbig(:,idir, isf, iphi, 1) = xdir(:,idir).*sf(:,isf).*xphase(:,iphi);
                end
            end
        else
            Xbig(:,idir, :, 1,1) = xdir(:,idir).*sf;
        end 
    end
end

Xbig = reshape(Xbig, NT, []);

if opts.use_derivative
    Xbig = max(filter([1; -1], 1, Xbig), 0);
end


% time embedding
stim_dur = ceil((opts.stim_dur)/Stim.bin_size);
if ~isfield(opts, 'stim_ctrs')
    opts.stim_ctrs = [0:10 15:5:stim_dur-2];
end
delta = opts.stim_ctrs(end) - opts.stim_ctrs(end-1);
xax = 0:stim_dur+delta;
Bt = tent_basis(xax, opts.stim_ctrs);

figure(1); clf, 
subplot(1,2,1)
plot(xax*Stim.bin_size, Bt)
title('Stimulus Temporal Basis')
xlabel('Time (bins)')

subplot(1,2,2)
imagesc(Bt)
title('Stimulus Temporal Basis')
xlabel('Basis Id')
ylabel('Time (bins)')

opts.nlags = size(Bt,2);
Xstim = temporalBases_dense(Xbig, Bt);

X = [X {Xstim}];
Labels = [Labels, {'Stim'}];

% stim onset
if opts.include_onset
    if opts.use_onset_only
        X = [X {temporalBases_dense(stim_on, Bt)}];
    else
        X = [X {temporalBases_dense(max(filter([1;-1], 1, stim_on), 0), Bt)}];
    end
    Labels = [Labels, {'Stim Onset'}];

end

% build running

Xdrift = tent_basis(1:NT, linspace(1, NT, opts.num_drift_tents));
X = [X {Xdrift}];
Labels = [Labels {'Drift'}];


opts.run_ctrs = linspace(opts.run_offset, opts.run_post, opts.nrunbasis);
run_basis = tent_basis(opts.run_offset:opts.run_post, opts.run_ctrs);

opts.sac_ctrs = linspace(opts.sac_offset, opts.sac_post, opts.nsacbasis);
sac_basis = tent_basis(opts.sac_offset:opts.sac_post, opts.sac_ctrs);

Xsac = zeros(NT, opts.nsacbasis);
Xrun = zeros(NT, opts.nrunbasis);

T = numel(Stim.trial_time);
num_trials = numel(Stim.grating_onsets);

for itrial = 1:num_trials
    % saccades
    if strcmp(opts.sac_covariate, 'onset')
        x = conv2(Stim.saccade_onset(:,itrial), sac_basis, 'full');
    else
        x = conv2(Stim.saccade_offset(:,itrial), sac_basis, 'full');
    end
    x = circshift(x, opts.sac_offset, 1);
    
    Xsac((itrial-1)*T + (1:T),:) = x(1:T,:);
    
    % running
    run_onset = max(filter([1; -1], 1, Stim.tread_speed(:,itrial)>opts.run_thresh), 0);
    x = conv2(run_onset, run_basis, 'full');
    x = circshift(x, opts.run_offset, 1);
    
    Xrun((itrial-1)*T + (1:T),:) = x(1:T,:);
    
end

X = [X {Xsac}];
Labels = [Labels, {'Saccade'}];

X = [X {Xrun}];
Labels = [Labels, {'Running'}];

% X = {Xstim, Xdrift, Xsac, Xrun};
opts.Labels = Labels; %{'Stim', 'Drift', 'Saccade', 'Running'};

