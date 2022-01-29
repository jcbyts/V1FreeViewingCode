
%%
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

%%

fout = '~/Data/Datasets/HuklabTreadmill/regression/';
flist = dir(fullfile(fout, '*.mat'));
{flist.name}'
ifile = 0;

%%
ifile = ifile + 1;
if ifile > numel(flist)
    ifile = 1;
end
load(fullfile(fout, flist(ifile).name));


model1 = 'nostim';
model2 = 'stimsac';

figure(1); clf
plot(Rpred.(model1).Rsquared, Rpred.(model2).Rsquared, '.'); hold on
plot(xlim, xlim ,'k')

figure(2); clf
plot(Rpred.(model1).Rpred')


%%
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

% Organize session data for analysis
Stim = convert_Dstruct_to_trials(D, 'bin_size', 1/60, 'pre_stim', .2, 'post_stim', .2);
Robs = bin_spikes_at_frames(Stim, D);

opts