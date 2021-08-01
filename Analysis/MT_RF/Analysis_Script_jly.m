%*********** ANALYSIS SCRIPT *******************************
%***********************************************************
%*** NOTE:  The goal here is just to analyze a single file
%***        Reads in a Exp struct created from the ImportScript
%***        that would be placed in the data folder, Exp.mat
%************************************************************


%% Load session
% get all sessions that have MT recordings from MarmoV5
sessList = io.dataFactory({'StimulusSuite', 'MarmoV5', 'Chamber', 'MT'});

for iSess = 2% 1:numel(sessList)
    try
        Exp = io.dataFactory(sessList{iSess});
    catch
        continue
    end
    
    %%
    
    % validTrials = io.getValidTrials(Exp, 'MTDotMapping');
    
    
    
    %% New Motion Analaysis of Space, with Sac and Motion Selectivity
    PROCESSED_DATA_DIR = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
    
    exname = strrep(Exp.FileTag, '.mat', '');
    SPClust = 1;  % need to revise to loop through all 64 units (and faster)
    GRID.box = [0,0,30,30];  % center x,y, width an d height
    GRID.div = 2.0;
    
    fname = sprintf('%s_%d_%d_%d_%d_%d.mat', exname, GRID.box(1), GRID.box(2), GRID.box(3), GRID.box(4), GRID.div);
    
    fout = fullfile(PROCESSED_DATA_DIR, fname);
    if exist(fout, 'file')==2
        load(fout)
    else
        
        for iTrial = 1:numel(Exp.D)
            
            vpxix = find(Exp.vpx.smo(:,1) > Exp.D{iTrial}.START_VPX & Exp.vpx.smo(:,1) < Exp.D{iTrial}.END_VPX);
            slistix = Exp.slist(:,1) > Exp.D{iTrial}.START_VPX & Exp.slist(:,1) < Exp.D{iTrial}.END_VPX;
            fprintf('%d) %d\n', iTrial, numel(vpxix))
            Exp.D{iTrial}.eyeSmo = Exp.vpx.smo(vpxix,:);
            Exp.D{iTrial}.slist = Exp.slist(slistix,:);
            Exp.D{iTrial}.slist(:,4:6) = Exp.D{iTrial}.slist(:,4:6) - vpxix(1) + 1;
        end
    
    
        [MoStimX,MoSacX,MoStimY] = Forage.StimMatrix_ForageMotionSpatialSacKernel(Exp,GRID);  %step 1
        save(fout, '-v7.3', 'MoStimX', 'MoSacX', 'MoStimY', 'GRID')
    end
    
end
%% copy to server
% server_string = 'jcbyts@sigurros';
% output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/grating_subspace';

server_string = 'jake@bancanus';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/MT_RF';

data_dir = PROCESSED_DATA_DIR;

command = 'scp ';
command = [command fullfile(data_dir, fname) ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)

%%
figure(1); clf
frameTime = MoStimX(:,1);

DensityCondition = unique(MoStimX(:,3));
Stim = MoStimX(:,4:end);

unique(Stim(:))

%% convert to dx dy

xax = -GRID.box(3)/2+1:GRID.div:GRID.box(3)/2 + GRID.box(1);
yax = -GRID.box(4)/2+1:GRID.div:GRID.box(4)/2 + GRID.box(2);

X = MoStimX(:,4:end);
dbin = 360/max(X(:));
NT = size(X,1);

ind = find(X~=0);
dx = zeros(size(X));
dy = zeros(size(X));

dx(ind) = cos(X(ind)*dbin/180*pi);
dy(ind) = sin(X(ind)*dbin/180*pi);

%% plot frame
figure(1); clf
vframes = find(sum(hypot(dx, dy),2) > 0);
i = i + 1; 
iframe = vframes(i);

% iframe = iframe + 1;
[xx,yy] = meshgrid(xax, yax);
quiver(xx(:), yy(:), dx(iframe,:)', dy(iframe,:)')

%%
num_lags = 10;
Xstim = makeStimRows([dx, dy], num_lags);

stas = (Xstim'*MoStimY);

%%
NC = size(MoStimY,2);

figure(clf)
cc = cc + 1;
if cc > NC
    cc = 1;
end
sta = stas(:,cc);
sta = sta ./ std(sta(:));
plot(sta); 
% title(cc)

sta = reshape(sta, [num_lags, size(xx).*[2 1]]);
figure(1); clf
for ilag = 1:num_lags
    subplot(1,num_lags, ilag)
%     quiver(xx(:), yy(:), squeeze(sta
    imagesc(squeeze(sta(ilag,:,:)), [-5 5])
end



%% *** MoStimY has all neural channels, Frames x N units, so loop through
%*** every unit and stack up all figures of them
%for k = 1:size(MoStimY,2)  
for k = [1,17,33,35,58,61,65,67,76,79]  % selection of some better units in Ellie_190120    
   ftag = sprintf('%s_u%d',exname,k); 
   if (sum(MoStimY(:,k)) > 1000)  % require at least 1000 spikes
     sum(MoStimY(:,k))
     rfinfo = Forage.ComputeForageMotionSpatialSacKernel(MoStimX,MoSacX,MoStimY(:,k),GRID,ftag);     % step 2, 
     Forage.PlotForageMotionSpatialSacKernel(rfinfo,ftag);
   else
     disp(sprintf('Too few spikes, skipping %s',ftag));
   end
end

