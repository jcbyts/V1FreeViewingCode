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


for id = 30:numel(sessList)
    sessionId = sessList{id};
%     try
            
    % load data
    Exp = io.dataFactoryTreadmill(sessionId);
    
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

            BIGROI = [-1 -1 1 1]*5;

            % eyePos = eyepos;
            eyePos = Exp.vpx.smo(:,2:3);
            % eyePos(:,1) = -eyePos(:,1);
            % eyePos(:,2) = -eyePos(:,2);

            stat = spat_rf_helper(Exp, 'ROI', BIGROI, ...
                'win', [0 12],...
                'binSize', .3, 'plot', true, 'debug', false, 'spikesmooth', 0);

            dataPath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
            exname = strrep(Exp.FileTag, '.mat', '');
            figDir = fullfile(dataPath, 'imported_sessions_qa', exname);

            NC = numel(stat.cgs);
            sx = ceil(sqrt(NC));
            sy = round(sqrt(NC));
            figure(11); clf
            ax = plot.tight_subplot(sx, sy, 0.02);
            for cc = 1:NC
                set(gcf, 'currentaxes', ax(cc))
                imagesc(stat.xax, stat.yax, stat.spatrf(:,:,cc));
                hold on
                plot(xlim, [0 0], 'r')
                plot([0 0], ylim, 'r')
                axis xy
                title(sprintf('Unit: %d', Exp.osp.cids(cc)))
            end


            plot.suplabel('Coarse RFs', 't');
            plot.fixfigure(gcf, 10, [sy sx]*2, 'offsetAxes', false)


        end


        % store RF information
        cids = unique(D.spikeIds);
        NC = numel(cids);
        assert(NC == numel(stat.rffit), 'unit count mismatch')

        for cc = 1:NC
            stat.rffit(cc).id = cids(cc);
            stat.rffit(cc).srf = stat.spatrf(:,:,cc);
            stat.rffit(cc).xax = stat.xax;
            stat.rffit(cc).yax = stat.yax;
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
    
    % save file
    fname = strrep(Exp.FileTag, '.mat', '_grat.mat');
    fdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');
    
    disp('Saving')
    save(fullfile(fdir, fname), '-v7.3', '-struct', 'D')
    save('Done')
%     catch me
%         fprintf('ERROR: [%s]\n', sessionId)
%     end
end




%% copy to server (for python analyses)
old_dir = pwd;

cd(fdir)
flist = dir(fullfile(fdir, '*.mat'));

server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/HuklabTreadmill/processed/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command '*.mat' ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)

cd(old_dir)

