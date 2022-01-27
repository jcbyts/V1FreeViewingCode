%% Step 0: set your paths
% The FREEVIEWING codebase uses matlab preferences to manage paths (so
% different users can have different paths)

addFreeViewingPaths('jakelaptop') % switch to your user
addpath Analysis/HukLabTreadmill/ % the code will always assume you're running from the FreeViewing base directory

%% Step 1: Make sure a session is imported
% You have to add the session to the datasets.xls file that is in the 
% google drive MANUALLY. If you open that file, you'll see the format for 
% everything

% dataFactoryTreadmill is the main workhorse for loading / importing data

% if you call it with no arguments, it will list all sessionIDs that have
% been added to the datasets.xls file
sessList = io.dataFactoryTreadmill({'StimulusSuite', 'MarmoV5', 'Chamber', 'V1'});

%% Step 1.1: Try importing a session


for id = 1:34
    sessionId = sessList{id};
    try
        
        % load data
        Exp = io.dataFactoryTreadmill(sessionId, 'abort_if_missing', true);
        
        % add spike times as fields
        Exp.spikeTimes = Exp.osp.st;
        Exp.spikeIds = Exp.osp.clu;
        
        if isfield(Exp, 'eyePos') && size(Exp.eyePos,2)==2 % pupil
            Exp.eyePos = [Exp.eyePos zeros(size(Exp.eyePos,1), 1)];
        end
        
        
        % convert to D struct format
        D = io.get_drifting_grating_output(Exp);
        
        if isfield(Exp, 'D')
            
            % get spatial RFs
            dotTrials = io.getValidTrials(Exp, 'Dots');
            if ~isempty(dotTrials)
                
                %             RFs = get_spatial_rfs(Exp);
                BIGROI = [-30 -10 30 10];
                
                eyePos = Exp.vpx.smo(:,2:3);
                
                binSize = .5;
                RFs = spat_rf_helper(Exp, 'ROI', BIGROI, ...
                    'win', [-5 12],...
                    'binSize', binSize, 'plot', false, 'debug', false, 'spikesmooth', 0);
                
                
                % PLOT RFS
                hasrf = find(RFs.area > binSize & RFs.maxV > 5);
                
                fprintf('%d / %d units have RFs\n', numel(hasrf), numel(RFs.cids))
                N = numel(hasrf);
                sx = round(sqrt(N));
                sy = ceil(sqrt(N));
                dims = size(RFs.spatrfs);
                
                figure(3); clf
                set(gcf, 'Color', 'w')
                ax = plot.tight_subplot(sx, sy, 0.01, 0.05);
                
                for cc = 1:(sx*sy)
                    set(gcf, 'currentaxes', ax(cc))
                    if cc > N
                        axis off
                        continue
                    end
                    
                    I = reshape(RFs.spatrfs(:,:,hasrf(cc)), [], 1);
                    I = reshape(zscore(I), dims(1:2));
                    
                    imagesc(RFs.xax, RFs.yax, I); hold on
                    try
                        plot(RFs.contours{hasrf(cc)}(:,1), RFs.contours{hasrf(cc)}(:,2), 'k')
                    end
                    axis xy
                    grid on
                    
                end
                
                colormap(plot.coolwarm)
                drawnow
                
            end
            
            
            % store RF information
            cids = unique(D.spikeIds);
            NC = numel(cids);
            stat = struct();
            for cc = 1:NC
                stat.rffit(cc).id = cids(cc);
                stat.rffit(cc).srf = nan;
                stat.rffit(cc).xax = nan;
                stat.rffit(cc).yax = nan;
                stat.rffit(cc).contour = [nan nan];
                stat.rffit(cc).area = nan;
                stat.rffit(cc).maxV = nan;
                stat.rffit(cc).center = nan;
            end
            
            for cc = RFs.cids(:)'
                unit = find(RFs.cids==cc);
                stat.rffit(cc).id = cc;
                stat.rffit(cc).srf = RFs.spatrfs(:,:,unit);
                stat.rffit(cc).xax = RFs.xax;
                stat.rffit(cc).yax = RFs.yax;
                stat.rffit(cc).contour = RFs.contours{unit};
                stat.rffit(cc).area = RFs.area(unit);
                stat.rffit(cc).maxV = RFs.maxV(unit);
                stat.rffit(cc).center = RFs.center(unit,:);
            end
        else
            % store empty RF information
            cids = unique(D.spikeIds);
            NC = numel(cids);
            %         stat.rffit = repmat(struct('id', [], 'srf', [], 'xax', [], 'yax', []), NC, 1);
            for cc = 1:NC
                stat.rffit(cc).id = cids(cc);
                stat.rffit(cc).srf = nan;
                stat.rffit(cc).xax = nan;
                stat.rffit(cc).yax = nan;
            end
        end
        
        D.units = stat.rffit;
        D.screen_bounds = reshape(kron((Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg, [-1; 1])', 1, []);
        % save file
        fname = strrep(Exp.FileTag, '.mat', '_grat.mat');
        fdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');
        
        disp('Saving')
        save(fullfile(fdir, fname), '-v7.3', '-struct', 'D')
        disp('Done')
    catch me
        fprintf('ERROR: [%s]\n', sessionId)
    end
end




%% copy to server (for python analyses)
old_dir = pwd;

cd(fdir)

server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/HuklabTreadmill/processed/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');

command = 'scp ';
command = [command '*.mat' ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)

cd(old_dir)

%% run regression analysis
fdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');
fout = strrep(fdir, 'gratings', 'regression');

flist = dir(fullfile(fdir, '*.mat'));
{flist.name}'
%%

for ifile = 20:numel(flist)
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
    
    if isfield(D, 'unit_area')
        if ~any(strcmp(D.unit_area, 'VISp'))
            continue
        end
    end
    
    
%     try
        [Stim, opts, Rpred, Running] = do_regression_analysis(D);
        
        disp('Saving...')
        save(fullfile(fout, flist(ifile).name), '-v7.3', 'Stim', 'opts', 'Rpred', 'Running')
        disp('Done')
%     end
end

%% fix D if it was created by scipy
ifile = 24;
ifile = 58;



