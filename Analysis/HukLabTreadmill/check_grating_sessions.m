addpath(genpath('MatlabCode'))
addpath Analysis/HukLabTreadmill/

fdir = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
if exist(fullfile(fdir, 'gratings'), 'dir')
    fdir = fullfile(fdir, 'gratings');
end

% fdir = '~/Data/Datasets/HuklabTreadmill/Dstruct/';
flist = dir(fullfile(fdir, '*.mat'));
{flist.name}'
ifile = 0;
%% load data
% ifile = ifile + 1;
for ifile = 44% 23:numel(flist)

fprintf('Running analysis for [%s]\n', flist(ifile).name)
    
% load Dstruct file
D = load(fullfile(fdir, flist(ifile).name));

% fix any wierdness from scipy
fields = fieldnames(D);
for f = 1:numel(fields)
    fprintf('[%s]\n', fields{f})
    if strcmp(fields{f}, 'unit_area')
        sz = size(D.unit_area);
        unit_area = cell(sz(1), 1);
        for i = 1:sz(1)
            unit_area{i} = strrep(D.unit_area(i,:), ' ', '');
        end
        D.unit_area = unit_area;
        continue

    end

    if iscell(D.(fields{f}))
        D.(fields{f}) = cell2mat(D.(fields{f}));
    end
end

if isempty(D.GratingContrast)
    continue
end
% Organize session data for analysis
Stim = convert_Dstruct_to_trials(D, 'bin_size', 1/60, 'pre_stim', .2, 'post_stim', .2);
if isempty(Stim)
    continue
end
Robs = bin_spikes_at_frames(Stim, D);
end

%%

Robs = filtfilt(ones(2,1)/2, 1, Robs);

U = eye(NC);

% build design matrix
opts = struct();
% opts.run_offset = -240;
% opts.run_post = 240;
% opts.nrunbasis = 400;

opts.stim_dur = median(D.GratingOffsets-D.GratingOnsets) + 0.05; % length of stimulus kernel
[X, opts] = build_design_matrix(Stim, opts);
%%
tread_speed = Stim.tread_speed(:);
pupil = Stim.eye_pupil(:);


tread_speed = tread_speed / nanstd(tread_speed); %#ok<*NANSTD> 
tread_speed = tread_speed - nanmean(tread_speed);

pupil = pupil / nanstd(pupil); %#ok<*NANSTD> 
pupil = pupil - nanmean(pupil);


figure(1); clf
plot(tread_speed)
hold on
r = imgaussfilt(zscore(mean(Robs,2)), 21);
plot(r);

good_inds = find(~isnan(tread_speed));

rta = (X{strcmp(opts.Labels, 'Running')}(good_inds,:)'*Robs(good_inds,:));
rta = rta./sum(X{strcmp(opts.Labels, 'Running')}(good_inds,:))';



figure(2); clf
plot(opts.run_ctrs, rta)
cc = 0
%%
cc = cc + 1;
if cc > size(Robs,2)
    cc = 1;
end



figure(1); clf
plot(imgaussfilt(tread_speed, 21))
hold on
r = imgaussfilt(zscore(Robs(:,cc)), 21);
plot(r, 'k');
title(cc)


figure(2); clf
plot(opts.run_ctrs, rta(:,cc))

%%
fout = '~/Data/Datasets/HuklabTreadmill/regression/';

parfor i = 44%6:numel(flist)
    
    run_analyses(fdir, fout, flist, i);

end

    

