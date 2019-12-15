

classdef sacGUI < handle
    % GUI for saccade labeling
    %
    % Purpose is to allow manual labeling of eye traces
    %
    % Initialization from Engbert, R., & Mergenthaler, K. (2006)
    % GUI by J.L. Yates
    
    properties (Access = public)
        H % handles
        X % X position
        Y % Y position
        pupil % pupil area
        time % time points
        P1
        P4
        Labels % Drift, Saccade, Blink, Lost Track
        xlim
        ylim
        sacid
        plotid
        slist
        window
        selected = []
    end
    
    methods (Access = public)
        function obj = sacGUI(fig)
            
            if nargin < 1
                fig = figure(1029321); % ks uses some defined figure numbers for plotting - with this random number we don't clash
                set(fig,'Name', 'sacGUI',...
                    'MenuBar', 'none',...
                    'Toolbar', 'none',...
                    'NumberTitle', 'off',...
                    'Units', 'normalized',...
                    'OuterPosition', [0.1 0.1 0.8 0.8]);
            end
            
            % check that required functions are present
            if ~exist('uiextras.HBox') %#ok<EXIST>
                error('sacGUI:init:uix', 'You must have the "uiextras" toolbox to use this GUI. Choose Home->Add-Ons->Get Add-ons and search for "GUI Layout Toolbox" by David Sampson. You may have to search for the author''s name to find the right one for some reason. If you cannot find it, go here to download: https://www.mathworks.com/matlabcentral/fileexchange/47982-gui-layout-toolbox\n')
            end

            
            obj.build(fig);
            
        end
        
        function initialize(obj)
            
            nTimePoints = numel(obj.X);
            obj.window =  -200:200;
            %******** process eye position to smooth and comp vel
            Xfilt = medfilt1(obj.X, 5);
            Yfilt = medfilt1(obj.Y, 5);
%             Xfilt = sgolayfilt(obj.X, 1, 7);
%             Yfilt = sgolayfilt(obj.Y, 1, 7);
            vx = [0; diff(Xfilt)];
            vy = [0; diff(Yfilt)];
            
            % convert to d.v.a / sec
            Fs = median(diff(obj.time));
            vx = vx / Fs;
            vy = vy / Fs;
            
            eyeSmo = [obj.time,Xfilt,Yfilt,obj.pupil, vx, vy, hypot(vx, vy)];
            
            
            if any(isnan(eyeSmo(1,:)))
                eyeSmo(1,:) = zeros(1,7);
            end
        
            for i = 1:size(eyeSmo,2)
                try
                    eyeSmo(:,i) = repnan(eyeSmo(:,i), 'nearest');
                end
            end
            
            fprintf(1, 'Initializing Labels\n')
            if exist(obj.H.Lfilename, 'file')
                obj.Labels = csvread(obj.H.Lfilename);
                saccades = obj.Labels==2;
                sacon = find(diff(saccades)==1);
                sacoff = find(diff(saccades)==-1);
                if saccades(1)
                    sacon = [1; sacon];
                end
                
                if saccades(end)
                    sacoff = [sacoff; numel(saccades)];
                end
                assert(numel(sacon)==numel(sacoff), 'why')
                saconoff = [sacon sacoff];
                
                nSaccades = size(saconoff,1);
                obj.slist = [zeros(nSaccades, 3) saconoff];
            else
                obj.Labels = zeros(nTimePoints, 1);
                obj.Labels(:) = 1; % initialize everything to fixation

                obj.slist = +saccadeflag.flag_saccades(eyeSmo, 'velThresh', 10, ...
                    'VFactor', 5, ...
                    'MinDuration', 20, ...
                    'MinGap', 10, ...
                    'FlagCurve', 1.2, ...
                    'SampRate', 1/Fs);
                nSaccades = size(obj.slist,1);
                fprintf(1, 'Found %d saccades\n', nSaccades);
                for iSac = 1:nSaccades
                    ix = obj.slist(iSac,4):obj.slist(iSac,5);
                    obj.Labels(ix) = 2;
                end
            end
            obj.sacid = 1;
            obj.plotid = obj.slist(obj.sacid,4);
            
            plot_sac_labels(obj)
             
        end
        
        
        function plot_sac_labels(obj)
            
            
                ix =  obj.plotid + obj.window;
                ix(ix < 1) = [];
                ix(ix > numel(obj.X)) = [];
                
                x = obj.X(ix);
                y = obj.Y(ix);
                t = obj.time(ix);
                L = obj.Labels(ix);
                
                
            if isempty(obj.H.hx)
                cmap = lines(4);
                for iLabel = 1:4
                    iix = L(:)==iLabel;
                    
                    if ~any(iix)
                        obj.H.hx(iLabel) = plot(obj.H.xpos, nan, nan, 'o', 'Color', cmap(iLabel,:), 'MarkerSize', 2);
                        obj.H.hy(iLabel) = plot(obj.H.ypos, nan, nan, 'o', 'Color', cmap(iLabel,:), 'MarkerSize', 2);
                    else
                        obj.H.hx(iLabel) = plot(obj.H.xpos, t(iix), x(iix), 'o', 'Color', cmap(iLabel,:), 'MarkerSize', 2);
                        obj.H.hy(iLabel) = plot(obj.H.ypos, t(iix), y(iix), 'o', 'Color', cmap(iLabel,:), 'MarkerSize', 2);
                    end
                    hold(obj.H.xpos, 'on')
                    hold(obj.H.ypos, 'on')
                end

            else
                
                for iLabel = 1:4
                    iix = L(:)==iLabel;
                    if ~any(iix)
                        set(obj.H.hx(iLabel), 'XData', nan);
                        set(obj.H.hx(iLabel), 'YData', nan);
                        set(obj.H.hy(iLabel), 'XData', nan);
                        set(obj.H.hy(iLabel), 'YData', nan);
                    else
                        set(obj.H.hx(iLabel), 'XData', t(iix));
                        set(obj.H.hx(iLabel), 'YData', x(iix));
                        set(obj.H.hy(iLabel), 'XData', t(iix));
                        set(obj.H.hy(iLabel), 'YData', y(iix));
                    end
                end
                
            end
            warning off
%             legend(obj.H.ypos, obj.H.hy, {'Fixation', 'Saccade', 'Blink', 'Lost Track'}, 'Location', 'NorthEast')
            warning on
            
            if isempty(obj.H.selected)
                
                if isempty(obj.selected)
                    obj.H.selected = plot(obj.H.xpos, nan, nan, 'r+');
                else
                    obj.H.selected = plot(obj.H.xpos, obj.time(obj.selected), obj.X(obj.selected), 'r+');
                end
            else
                if isempty(obj.selected)
                    set(obj.H.selected, 'XData', nan, 'YData', nan);
                else
                    set(obj.H.selected, 'XData', obj.time(obj.selected), 'YData', obj.X(obj.selected));
                end
                
            end
            
        end
        
        
        function build(obj, f)
            % construct the GUI with appropriate panels
            obj.H.fig = f;
            set(f, 'UserData', obj);
%             
            set(f, 'KeyPressFcn', @(f,k)obj.keyboardFcn(f, k));
            
            obj.H.Plots = uiextras.VBox('Parent', f,...
                'DeleteFcn', @(~,~)obj.cleanup(), 'Visible', 'on', ...
                'Spacing', 2, 'Padding', 3);
            
            
            % --- plotting axes
            obj.H.xpos = axes( 'Parent', obj.H.Plots );
            obj.H.ypos = axes( 'Parent', obj.H.Plots );
            
            obj.H.Buttons = uiextras.HBox( 'Parent', obj.H.Plots);
            
            set(obj.H.Plots, 'Heights', [-5; -5; -1])
            
            title(obj.H.xpos, 'X position')
            title(obj.H.ypos, 'Y position')
            
            xlabel(obj.H.xpos, 'Time (ms)')
            xlabel(obj.H.ypos, 'Time (ms)')
            
            ylabel(obj.H.xpos, 'Degrees')
            ylabel(obj.H.ypos, 'Degrees')
            
            % --- button panels
            obj.H.labelPanel = uiextras.BoxPanel( 'Parent', obj.H.Buttons, 'Title', 'Label');
            obj.H.navigatePanel = uiextras.BoxPanel( 'Parent', obj.H.Buttons, 'Title', 'Navigate');
            obj.H.filePanel = uiextras.BoxPanel( 'Parent', obj.H.Buttons, 'Title', 'File');
            
            set(obj.H.Buttons, 'Widths', [-2; -2; -1])
            
            % --- button boxes
            obj.H.fileButtonBox = uiextras.HButtonBox( 'Parent', obj.H.filePanel);
            obj.H.labelButtonBox = uiextras.HButtonBox( 'Parent', obj.H.labelPanel);
            obj.H.navigateButtonBox = uiextras.HButtonBox( 'Parent', obj.H.navigatePanel);
            
            % --- individual buttons
            obj.H.labelSaccade = uicontrol( 'Parent', obj.H.labelButtonBox, ...
                    'String', 'Saccade', ...
                    'Callback', @(~,~)obj.labelSaccade);
            
            obj.H.labelFixation = uicontrol( 'Parent', obj.H.labelButtonBox, ...
                    'String', 'Fixation', ...
                    'Callback', @(~,~)obj.labelFixation);
                
            obj.H.labelBlink = uicontrol( 'Parent', obj.H.labelButtonBox, ...
                    'String', 'Blink', ...
                    'Callback', @(~,~)obj.labelBlink);
                
            obj.H.labelLost = uicontrol( 'Parent', obj.H.labelButtonBox, ...
                    'String', 'Lost Track', ...
                    'Callback', @(~,~)obj.labelLost);
                
            obj.H.previous = uicontrol( 'Parent', obj.H.navigateButtonBox, ...
                    'String', 'Previous', ...
                    'Callback', @(~,~)obj.prevSaccade);
                
            obj.H.next = uicontrol( 'Parent', obj.H.navigateButtonBox, ...
                    'String', 'Next', ...
                    'Callback', @(~,~)obj.nextSaccade);
                
            obj.H.grab = uicontrol( 'Parent', obj.H.navigateButtonBox, ...
                    'String', 'Grab Points', ...
                    'Callback', @(~,~)obj.grabPoints);
                
            obj.H.load = uicontrol( 'Parent', obj.H.fileButtonBox, ...
                    'String', 'Load', ...
                    'Callback', @(~,~)obj.selectFileDlg);
                
            obj.H.save = uicontrol( 'Parent', obj.H.fileButtonBox, ...
                    'String', 'Save', ...
                    'Callback', @(~,~)obj.saveFile);
                
            % --- Data traces initialize
            obj.H.hx = [];
            obj.H.hy = [];
            obj.H.selected = [];
            
        end
        
        
        function selectFileDlg(obj)
            [filename, pathname] = uigetfile('*.*', 'Pick a data file.');
            
            if filename==0 % 0 when cancel
                fprintf(1, 'File load cancelled by user')
                return
            end
            
            obj.H.filename = fullfile(pathname, filename);
            fprintf('Loading [%s] \n', filename)
            Exp = load(obj.H.filename);
            
            obj.H.Xfilename = strrep(obj.H.filename, '.mat', 'X.csv');
            obj.H.Yfilename = strrep(obj.H.filename, '.mat', 'Y.csv');
            obj.H.Lfilename = strrep(obj.H.filename, '.mat', 'L.csv');
            
            % convert eye position to degrees
            nTrials = numel(Exp.D);
            validTrials = 1:nTrials;
            
            % gain and offsets from online calibration
            cx = cellfun(@(x) x.c(1), Exp.D(validTrials));
            cy = cellfun(@(x) x.c(2), Exp.D(validTrials));
            dx = cellfun(@(x) x.dx, Exp.D(validTrials));
            dy = cellfun(@(x) x.dy, Exp.D(validTrials));
            
            % use the most common value across trials (we should've only calibrated
            % once in these sessions)
            cx = mode(cx);
            cy = mode(cy);
            dx = mode(dx);
            dy = mode(dy);
            
            % x and y position
            vxx = Exp.vpx.raw(:,2);
            vyy = Exp.vpx.raw(:,3);
            
            % convert to d.v.a.
            vxx = (vxx - cx)/(dx * Exp.S.pixPerDeg);
            vyy = 1 - vyy;
            vyy = (vyy - cy)/(dy * Exp.S.pixPerDeg);

            obj.X = vxx;
            obj.Y = vyy;
            obj.pupil = Exp.vpx.raw(:,4);
            obj.time = Exp.vpx.raw(:,1);
            
            obj.xlim = [0 10];
            obj.ylim = [-10 10];
            
            % initialize plots
            obj.window =  -200:200;
            obj.plotid = 1;
            obj.sacid = 1;
            
            if isfield(Exp, 'slist')
                obj.slist = Exp.slist;
                
                if isfield(Exp.vpx, 'Labels')
                    obj.Labels = Exp.vpx.Labels;
                else
                    nTimePoints = numel(obj.X);
                    obj.Labels = zeros(nTimePoints, 1);
                    obj.Labels(:) = 1; % initialize everything to fixation
                    
                    nSaccades = size(obj.slist,1);
                    fprintf(1, 'Found %d saccades\n', nSaccades);
                    for iSac = 1:nSaccades
                        ix = obj.slist(iSac,4):obj.slist(iSac,5);
                        obj.Labels(ix) = 2;
                    end
                end
                
                 obj.plotid = obj.slist(obj.sacid,4);
            
                plot_sac_labels(obj)
            else
                initialize(obj)
            end
        end

        
       
        function keyboardFcn(obj, ~, k)
            disp(k.Key)
            switch k.Key
                case 'rightarrow'
                    obj.plotid = min(obj.plotid + 50, numel(obj.X));
                    plot_sac_labels(obj);
                    
                case 'leftarrow'
                    obj.plotid = max(obj.plotid - 50,1);
                    plot_sac_labels(obj);
                    
                case 's'
                    labelSaccade(obj);
                case 'b'
                    labelBlink(obj);
                case 'g'
                    grabPoints(obj);
                case 'f'
                    labelFixation(obj)
                case 'l'
                    labelLost(obj)
            end

        end
            
        
      
        
        function cleanup(obj)
            fclose('all');
        end
        
    end
    
    % --- Callback methods
    methods (Access = public)
        
        function labelSaccade(obj)
            
            if ~isempty(obj.selected)
                obj.Labels(obj.selected) = 2;
            end
            
            % clear selected
            obj.selected = [];
            
            % update plots
            plot_sac_labels(obj)
            
        end
        
        function labelFixation(obj)
            
            if ~isempty(obj.selected)
                obj.Labels(obj.selected) = 1;
            end
            
            % clear selected
            obj.selected = [];
            
            % update plots
            plot_sac_labels(obj)
            
        end
        
        function labelBlink(obj)
            
            if ~isempty(obj.selected)
                obj.Labels(obj.selected) = 3;
            end
            
            % clear selected
            obj.selected = [];
            
            % update plots
            plot_sac_labels(obj)
            
        end
        
        function labelLost(obj)
            
            if ~isempty(obj.selected)
                obj.Labels(obj.selected) = 4;
            end
                
            % clear selected
            obj.selected = [];
            
            % update plots
            plot_sac_labels(obj)
            
        end
        
        function prevSaccade(obj)
            
            obj.plotid = max(obj.plotid - 50, 1);

            % clear selected
            obj.selected = [];
            
            % update plots
            plot_sac_labels(obj)
            
        end
        
        function nextSaccade(obj)
            obj.plotid = min(obj.plotid + 50, numel(obj.X));

            % clear selected
            obj.selected = [];
            
            % update plots
            plot_sac_labels(obj)
            
        end
        
        function grabPoints(obj)
            
            set(obj.H.fig, 'currentaxes', obj.H.xpos)
            
            xy = ginput(2);
            
            obj.selected = obj.time > xy(1,1) & obj.time < xy(2,1);
            
            % update plots
            plot_sac_labels(obj)
        end
        
        
        function saveFile(obj)
            
            fprintf(1, 'Saving data\n')
            Exp = load(obj.H.filename);
            Exp.vpx.Labels = obj.Labels;
            saccades = Exp.vpx.Labels(:) == 2;
            sstart = find(diff(saccades)==1);
            sstop = find(diff(saccades)==-1);
            if saccades(1)
                sstart = [1; sstart];
            end
            
            if saccades(end)
                sstop = [sstop; numel(saccades)];
            end
            
            nSaccades = numel(sstart);
            midpoint = zeros(nSaccades, 1);
            for iSaccade = 1:nSaccades
                [~, id] = max(Exp.vpx.smo(sstart(iSaccade):sstop(iSaccade),7));
                midpoint(iSaccade) = sstart(iSaccade) + id;
            end
            sl = [Exp.vpx.smo(sstart,1) Exp.vpx.smo(sstop,1) Exp.vpx.smo(midpoint,1) sstart sstop midpoint];
            Exp.slist = sl;
            save(obj.H.filename, '-v7.3', '-struct','Exp')
            
            csvwrite(obj.H.Xfilename, obj.X)
            csvwrite(obj.H.Yfilename, obj.Y)
            csvwrite(obj.H.Lfilename, obj.Labels)
            fprintf(1, 'Done\n')
            
        end
        
        
    end
    
    methods(Static)
        
       
        
    end
    
end


