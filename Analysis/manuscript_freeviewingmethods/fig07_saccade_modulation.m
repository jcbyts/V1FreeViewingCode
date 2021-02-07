
%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

addpath Analysis/202001_K99figs_01  
addpath Analysis/manuscript_freeviewingmethods/

%% Load analyses from prio steps

sesslist = io.dataFactoryGratingSubspace;
sesslist = sesslist(1:57); % exclude monash sessions

% Spatial RFs
sfname = fullfile('Data', 'spatialrfs.mat');
load(sfname)

% Grating RFs
gfname = fullfile('Data', 'gratrf.mat');
load(gfname)


%% get fftrf
% Analyze FFTRF
fftrf = cell(numel(sesslist),1);
fftname = fullfile('Data', 'fftrf.mat');
if exist(fftname, 'file')==2
    disp('Loading FFT RFs')
    load(fftname)
else
    
    %%
    for iEx = 1:numel(sesslist)
%         if isempty(fftrf{iEx})
            try
                
                try % use JRCLUST sorts if they exist
                    sorter = 'jrclustwf';
                    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                catch % otherwise, use Kilosort
                    sorter = 'kilowf';
                    Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                end
                
                evalc('rf_pre = fixrate_by_fftrf(Exp, Srf{iEx}, Sgt{iEx}, ''plot'', false, ''usestim'', ''pre'', ''alignto'', ''fixon'');');
                evalc('rf_post = fixrate_by_fftrf(Exp, Srf{iEx}, Sgt{iEx}, ''plot'', true, ''usestim'', ''post'', ''alignto'', ''fixon'');');
                
                fftrf{iEx} = struct();
                fftrf{iEx}.rfs_pre = rf_pre;
                fftrf{iEx}.rfs_post = rf_post;
                fftrf{iEx}.sorter = sorter;
            catch me
                disp('ERROR ERROR')
                disp(me.message)
            end
%         end
    end
    %%
    
    save(fftname, '-v7.3', 'fftrf')
end

%%
fields = {'rfs_post', 'rfs_pre'};
field = fields{1};
cmap = lines;

iEx = 46;
cc = cc + 1;

if cc > numel(fftrf{iEx}.(field))
    cc = 1;
end

figure(1); clf
subplot(3,2,1) % spatial RF
imagesc(Srf{iEx}.xax, Srf{iEx}.yax, Srf{iEx}.spatrf(:,:,cc)); axis xy
hold on
plot(fftrf{iEx}.(field)(cc).rfLocation(1), fftrf{iEx}.(field)(cc).rfLocation(2), 'or')

subplot(3,2,2) % grating fit
imagesc(fftrf{iEx}.(field)(cc).rf.kx, fftrf{iEx}.(field)(cc).rf.ky, fftrf{iEx}.(field)(cc).rf.Ifit)
title(cc)

for  f = 1:2
    field = fields{f};
    subplot(3,2,3+(f-1)*2) % X proj
    bar(fftrf{iEx}.(field)(cc).xproj.bins, fftrf{iEx}.(field)(cc).xproj.cnt, 'FaceColor', .5*[1 1 1]); hold on
    lev = fftrf{iEx}.(field)(cc).xproj.levels(1);
    iix = fftrf{iEx}.(field)(cc).xproj.bins <= lev;
    bar(fftrf{iEx}.(field)(cc).xproj.bins(iix), fftrf{iEx}.(field)(cc).xproj.cnt(iix), 'FaceColor', cmap(5,:));
    lev = fftrf{iEx}.(field)(cc).xproj.levels(2);
    iix = fftrf{iEx}.(field)(cc).xproj.bins >= lev;
    bar(fftrf{iEx}.(field)(cc).xproj.bins(iix), fftrf{iEx}.(field)(cc).xproj.cnt(iix), 'FaceColor', cmap(1,:));
    
    subplot(3,2,4+(f-1)*2) % PSTH
    mrate = fftrf{iEx}.(field)(cc).rateHi;
    srate = fftrf{iEx}.(field)(cc).stdHi / sqrt(fftrf{iEx}.(field)(cc).nHi);
    plot.errorbarFill(fftrf{iEx}.(field)(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:)); hold on
    mrate = fftrf{iEx}.(field)(cc).rateLow;
    srate = fftrf{iEx}.(field)(cc).stdLow / sqrt(fftrf{iEx}.(field)(cc).nLow);
    plot.errorbarFill(fftrf{iEx}.(field)(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:));
    xlim([-.05 .4])
end

plot.suplabel(strrep(sesslist{iEx}, '_', ' '), 't');


frf = fftrf{iEx}.rfs_post(cc).frf;
nsteps = numel(fftrf{iEx}.rfs_post(cc).frfsteps);
figure(10); clf
clim = [min(frf(:)) max(frf(:))];
for i = 1:nsteps
    subplot(1,nsteps+1, i)
    imagesc(fftrf{iEx}.(field)(cc).rf.kx, fftrf{iEx}.(field)(cc).rf.ky, frf(:,:,i), clim)
end

%%

field = fields{1};
fftrf{1}.(field)(cc).rateHi

%% Loop over and get relevant statistics

ar = []; % sqrt area (computed from gaussian fit)
ecc = []; % eccentricity
maxV = []; % volume of RF blob

sfPref = [];  % spatial frequency preference
sfBw = [];    % spatial frequency bandwidth (FWHM)
oriPref = []; % orientation preference
oriBw  = [];  % orientation bandwidth (FWHM)

sigg = [];
sigs = [];

r2 = [];   % r-squared from gaussian fit to RF
gtr2 = []; % r-squared of parametric fit to frequecy RF

ctr = []; % counter for tracking cell number
cgs = []; % cluster quality

mshift = []; % how much did the mean shift during fitting (measure of whether we got stuck in local minimum)

% FFT modulations
mrateHi = [];
mrateLow = [];

field = 'rfs_post';

zthresh = 8;
for ex = 1:numel(Srf)
    
    if ~isfield(Srf{ex}, 'rffit') || ~isfield(Sgt{ex}, 'rffit') || (numel(Sgt{ex}.rffit) ~= numel(Srf{ex}.rffit))
        continue
    end
    
    NC = numel(Srf{ex}.rffit);
    for cc = 1:NC
        if ~isfield(Srf{ex}.rffit(cc), 'mu')
            continue
        end
        
         mu = Srf{ex}.rffit(cc).mu;
         C = Srf{ex}.rffit(cc).C;
         maxV = [maxV; Srf{ex}.maxV(cc)];
         
         if isempty(Sgt{ex}.rffit(cc).r2) || isempty(Srf{ex}.rffit(cc).r2)
             oriPref = [oriPref; nan];
             oriBw = [oriBw; nan];
             sfPref = [sfPref; nan];
             sfBw = [sfBw; nan];
             gtr2 = [gtr2; nan];
             
             r2 = [r2; nan]; % store r-squared
             ar = [ar; nan];
             ecc = [ecc; nan];
             sigs = [sigs; false];
             sigg = [sigg; false];
             mus = [mus; nan(1,2)];
             Cs = [Cs; nan(1,4)];
             cgs = [cgs; nan];
             mshift = [mshift; nan];
             
             mrateHi = [mrateHi; nan(1,601)];
             mrateLow = [mrateLow; nan(1,601)];
             continue
         end
         
         
         oriPref = [oriPref; Sgt{ex}.rffit(cc).oriPref];
         oriBw = [oriBw; Sgt{ex}.rffit(cc).oriBandwidth];
         sfPref = [sfPref; Sgt{ex}.rffit(cc).sfPref];
         sfBw = [sfBw; Sgt{ex}.rffit(cc).sfBandwidth];
         gtr2 = [gtr2; Sgt{ex}.rffit(cc).r2];
             
         % significance
         zrf = Sgt{ex}.rf(:,:,cc)*Sgt{ex}.fs_stim / Sgt{ex}.sdbase(cc);
         z = reshape(zrf(Sgt{ex}.timeax>=0,:), [], 1);
         sigg = [sigg; sum(z > zthresh) > (1-normcdf(zthresh))];
         
         
         spatsig = sum(diff(Srf{ex}.rffit(cc).betaCi, [], 2));
         sigs = [sigs; spatsig];
        
         r2 = [r2; Srf{ex}.rffit(cc).r2]; % store r-squared
         ar = [ar; Srf{ex}.rffit(cc).ar];
         ecc = [ecc; Srf{ex}.rffit(cc).ecc];
        
         
         
         cgs = [cgs; Srf{ex}.cgs(cc)];
         mshift = [mshift; Srf{ex}.rffit(cc).mushift]; %#ok<*AGROW>
         
         % FFT stuff
         mrateHi = [mrateHi; fftrf{ex}.(field)(cc).rateHi];
         mrateLow = [mrateLow; fftrf{ex}.(field)(cc).rateLow];
         
         ctr = [ctr; [numel(r2) numel(gtr2) size(mrateHi,1)]];
         
         if ctr(end,1) ~= ctr(end,3)
             keyboard
         end
    end
end

% wrap orientation
oriPref(oriPref < 0) = 180 + oriPref(oriPref < 0);
oriPref(oriPref > 180) = oriPref(oriPref > 180) - 180;

[sum(sigs) sum(sigg)]

%%

nrateHi = mrateHi ./ max(mrateHi,2);
nrateLow = mrateLow ./ max(mrateHi,2);

ix = sigs & sigg;
% ix = ix & ecc <5;
figure(1); clf


imagesc( (nrateHi(ix,:) - nrateLow(ix,:)))

figure(2); clf
plot(nanmean(nrateHi(ix,:))); hold on
plot(nanmean(nrateLow(ix,:)))
%%