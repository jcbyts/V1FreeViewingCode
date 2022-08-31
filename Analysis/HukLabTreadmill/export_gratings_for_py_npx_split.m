%% Step 0: set your paths
% The FREEVIEWING codebase uses matlab preferences to manage paths (so
% different users can have different paths)

addFreeViewingPaths('jakelaptop') % switch to your user
addpath Analysis/HukLabTreadmill/ % the code will always assume you're running from the FreeViewing base directory

%% Step 1: Make sure a session is imported
% You have to add the session to the datasets.xls file that is in the 
% google drive MANUALLY. If you open that file, you'll see the format for 
% everything

sessList =   {'gru_20211217', 'gru_20220412'};
fovealDepths = {[1600 3200], [0 600]};
calcarineDepths = {[0 950], [800 2000]};
eyePosSign = {[1 1], [-1 -1]};

%% Step 1.1: Try importing a session


for id = 1:2
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
        
   
        
        % get spatial RFs
        dotTrials = io.getValidTrials(Exp, 'Dots');
        if ~isempty(dotTrials)
            
            %             RFs = get_spatial_rfs(Exp);
            BIGROI = [-30 -10 30 10];
            
            eyePos = Exp.vpx.smo(:,2:3);
            eyePos(:,1) = eyePosSign{id}(1)*eyePos(:,1);
            eyePos(:,2) = eyePosSign{id}(2)*eyePos(:,2);
            
            binSize = .5;
            RFs = spat_rf_helper(Exp, 'ROI', BIGROI, ...
                'win', [-5 12],...
                'eyePos', eyePos, ...
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
        
        %%
        fovealCids = Exp.osp.cids(Exp.osp.clusterDepths > fovealDepths{id}(1) & Exp.osp.clusterDepths < fovealDepths{id}(2));
        calcarineCids = Exp.osp.cids(Exp.osp.clusterDepths > calcarineDepths{id}(1) & Exp.osp.clusterDepths < calcarineDepths{id}(2));
        
        figure(1); clf
        for i = 1:numel(RFs.cids)
            cc = RFs.cids(i);
            if isempty(RFs.contours{i})
                continue
            end
            x = RFs.contours{i}(:,1);
            y = RFs.contours{i}(:,2);
            ind = find(Exp.osp.cids==cc);
            depth = Exp.osp.clusterDepths(ind);
            
            if ismember(cc, fovealCids)
                plot3(x, y, ones(size(x))*depth, 'r'); hold on
            elseif ismember(cc, calcarineCids)
                plot3(x, y, ones(size(x))*depth, 'b'); hold on
            else
                plot3(x, y, ones(size(x))*depth, 'k'); hold on
            end

        end

        %%
        % store RF information
        cids = fovealCids;
        NC = numel(cids);
        statFoveal = struct();
        for cc = 1:NC
            statFoveal.rffit(cc).id = cids(cc);
            statFoveal.rffit(cc).srf = nan;
            statFoveal.rffit(cc).xax = nan;
            statFoveal.rffit(cc).yax = nan;
            statFoveal.rffit(cc).contour = [nan nan];
            statFoveal.rffit(cc).area = nan;
            statFoveal.rffit(cc).maxV = nan;
            statFoveal.rffit(cc).center = nan;
        end
        
        rfcids = intersect(RFs.cids, fovealCids);
        for cid = rfcids(:)'
            unit = find(RFs.cids==cid);
            cc = find([statFoveal.rffit.id]==cid);
            statFoveal.rffit(cc).srf = RFs.spatrfs(:,:,unit);
            statFoveal.rffit(cc).xax = RFs.xax;
            statFoveal.rffit(cc).yax = RFs.yax;
            statFoveal.rffit(cc).contour = RFs.contours{unit};
            statFoveal.rffit(cc).area = RFs.area(unit);
            statFoveal.rffit(cc).maxV = RFs.maxV(unit);
            statFoveal.rffit(cc).center = RFs.center(unit,:);
        end
        

        % store RF information
        cids = calcarineCids;
        NC = numel(cids);
        statCalc = struct();
        for cc = 1:NC
            statCalc.rffit(cc).id = cids(cc);
            statCalc.rffit(cc).srf = nan;
            statCalc.rffit(cc).xax = nan;
            statCalc.rffit(cc).yax = nan;
            statCalc.rffit(cc).contour = [nan nan];
            statCalc.rffit(cc).area = nan;
            statCalc.rffit(cc).maxV = nan;
            statCalc.rffit(cc).center = nan;
        end
        
        rfcids = intersect(RFs.cids, calcarineCids);
        for cid = rfcids(:)'
            unit = find(RFs.cids==cid);
            cc = find([statCalc.rffit.id]==cid);
            statCalc.rffit(cc).srf = RFs.spatrfs(:,:,unit);
            statCalc.rffit(cc).xax = RFs.xax;
            statCalc.rffit(cc).yax = RFs.yax;
            statCalc.rffit(cc).contour = RFs.contours{unit};
            statCalc.rffit(cc).area = RFs.area(unit);
            statCalc.rffit(cc).maxV = RFs.maxV(unit);
            statCalc.rffit(cc).center = RFs.center(unit,:);
        end

        % SAVE FOVEAL GROUP
        Dfov = D;
        iix = ismember(D.spikeIds, fovealCids);
        Dfov.spikeTimes = Dfov.spikeTimes(iix);
        Dfov.spikeIds = Dfov.spikeIds(iix);
        Dfov.units = statFoveal.rffit;
        Dfov.screen_bounds = reshape(kron((Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg, [-1; 1])', 1, []);

        fname = [sessList{id} 'fov_grat.mat'];
        fdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');
        
        disp('Saving foveal group')
        save(fullfile(fdir, fname), '-v7.3', '-struct', 'Dfov')
        disp('Done')

        % SAVE CALCARINE GROUP
        Dcalc = D;
        iix = ismember(D.spikeIds, calcarineCids);
        Dcalc.spikeTimes = Dcalc.spikeTimes(iix);
        Dcalc.spikeIds = Dcalc.spikeIds(iix);
        Dcalc.units = statCalc.rffit;
        Dcalc.screen_bounds = reshape(kron((Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg, [-1; 1])', 1, []);
        
        fname = [sessList{id} 'calc_grat.mat'];
        fdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');
        
        disp('Saving calcarine group')
        save(fullfile(fdir, fname), '-v7.3', '-struct', 'Dcalc')
        disp('Done')

        %%

        cids = unique(Dfov.spikeIds);
        NC = numel(cids);
        assert(NC==numel(Dfov.units), 'import_supersession: mismatch in number of units on session')

        cids = unique(Dcalc.spikeIds);
        NC = numel(cids);
        assert(NC==numel(Dcalc.units), 'import_supersession: mismatch in number of units on session')

        %%
    catch me
        fprintf('ERROR: [%s]\n', sessionId)
    end
end




%% copy to server (for python analyses)
old_dir = pwd;

cd(fdir)
flist = dir(fullfile(fdir, '*fov*.mat'));
flist = [flist; dir(fullfile(fdir, '*calc*.mat'))];

server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/HuklabTreadmill/gratings/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');

command = 'scp ';
for ifile = 1:numel(flist)
    command = [command flist(ifile).name ' '];
end
command = [command server_string ':' output_dir];

system(command)


cd(old_dir)

%% copy to server (for python analyses)
old_dir = pwd;

cd(fdir)
flist = dir(fullfile(fdir, '*_all*.mat'));

server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/HuklabTreadmill/gratings/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');

command = 'scp ';
for ifile = 1:numel(flist)
    command = [command flist(ifile).name ' '];
end
command = [command server_string ':' output_dir];

system(command)
% disp(command)

cd(old_dir)


%%
% %% run regression analysis
% fdir = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');
% fout = strrep(fdir, 'gratings', 'regression');
% 
% flist = dir(fullfile(fdir, '*fov*.mat'));
% flist = [flist; dir(fullfile(fdir, '*calc*.mat'))];
% 
% {flist.name}'





