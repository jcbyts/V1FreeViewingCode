

%%
% dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
% 
% for i = 44:57
%     disp(i)
%     Exp = io.dataFactoryGratingSubspace(i);
% %   Exp.vpx.raw = unique(Exp.vpx.raw, 'rows');
% %   Exp = saccadeflag.run_saccade_detection(Exp, 'ShowTrials', false);
%     fname = fullfile(dataPath, Exp.FileTag);
%     save(fname,'-v7.3', '-struct', 'Exp');
% end


%%

Exp = io.dataFactoryGratingSubspace(5);
Exp.vpx.raw(:,3) = -Exp.vpx.raw(:,3);
[X, Robs, opts] = io.preprocess_spatialmapping_data(Exp, 'ROI', [-300 -200 300 200]*2);

%%
nlags = 12;
NC = size(Robs,2);

y = Robs - mean(Robs);
ybar = mean(y,2);
for cc = 1:NC
    y(:,cc) = ybar;
end
% NC = 1;
stas = zeros(prod(opts.dims), nlags, NC);
for ilag = 1:nlags
    lag = ilag;
    fprintf('Lag %d/%d\n', ilag, nlags)
    
    sta = (X(1:end-lag+1,:))'*y(lag:end, :);
    stas(:,ilag,:) = sta ./ sum(X)';
    
end


%
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

figure(1); clf
for kNeuron = 1%:NC
    fprintf('Unit %d/%d\n', kNeuron, NC)
%     y = ytmp(:,kNeuron)-mean(ytmp(:,kNeuron)); % subtract mean to account for DC
%     sta = simpleRevcorr(X, y, nlags);
    sta = stas(:,:,kNeuron)';
    subplot(sx, sy, kNeuron, 'align')
    plot(sta)

    [u,s,v] = svd(sta);
    sd = sign(sum(u(:,1)));
    rfs(kNeuron).separability = s(1) / sum(diag(s));
    rfs(kNeuron).sta = sta;
    rfs(kNeuron).trf = u(:,1)*sd;
    rfs(kNeuron).srf = reshape(v(:,1)*sd, opts.dims);
    
end


figure(2); clf


for kNeuron = 1%:NC
    imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, rfs(kNeuron).srf)
    drawnow
end


%%
% dataPath = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
% 
% for i = 1:27
%     disp(i)
%     Exp = io.dataFactoryGratingSubspace(i);
%     Exp.vpx.raw(:,3) = -Exp.vpx.raw(:,3);
%     Exp = saccadeflag.run_saccade_detection(Exp, 'ShowTrials', false);
%     fname = fullfile(dataPath, Exp.FileTag);
%     save(fname,'-v7.3', '-struct', 'Exp');
% end

%% 



%%
sta = (zscore(stimX)'*Robs); % ./ sum(stimX)';
NC = size(sta,2);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(1); clf
for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    I = reshape(sta(:,cc), opts.dims);
    imagesc(opts.xax, opts.yax, I)
end

%%
figure(1); clf
plot(opts.xPosition, opts.yPosition, '.')

Exp = io.dataFactoryGratingSubspace


%%

% load dataset
kEx = kEx + 1;
Exp = io.dataFactoryGratingSubspace(kEx);

% try
%

% --- find valid trials
validTrials = intersect(io.getValidTrials(Exp, 'BigDots'), io.getValidTrials(Exp, 'Ephys'));
numValidTrials = numel(validTrials);

fprintf('Found %d valid trials\n', numValidTrials)

% Eye calibration
cx = mode(cellfun(@(x) x.c(1), Exp.D(validTrials)));
cy = mode(cellfun(@(x) x.c(2), Exp.D(validTrials)));
dx = mode(cellfun(@(x) x.dx, Exp.D(validTrials)));
dy = mode(cellfun(@(x) x.dy, Exp.D(validTrials)));


% Extract trial-specific values
frameTimes = cellfun(@(x) x.PR.NoiseHistory(:,1), Exp.D(validTrials(:)), 'uni', 0);

xpos = cellfun(@(x) x.PR.NoiseHistory(:,1+(1:x.PR.noiseNum)), Exp.D(validTrials(:)), 'uni', 0);
ypos = cellfun(@(x) x.PR.NoiseHistory(:,x.PR.noiseNum+1+(1:x.PR.noiseNum)), Exp.D(validTrials(:)), 'uni', 0);

% check if two conditions were run a
nd = cellfun(@(x) size(x,2), xpos);
xpos = cell2mat(xpos(nd == max(nd)));
ypos = cell2mat(ypos(nd == max(nd)));
frameTimes = Exp.ptb2Ephys(cell2mat(frameTimes(nd==max(nd))));



% convert to d.v.a.
eyeDat = unique(Exp.vpx.raw(:,1:3), 'rows');
eyeDat(:,2) = (eyeDat(:,2) - cx)/(dx * Exp.S.pixPerDeg);
eyeDat(:,3) = 1 - eyeDat(:,3);
eyeDat(:,3) = (eyeDat(:,3) - cy)/(dy * Exp.S.pixPerDeg);
% convert to pixels
eyeDat(:,2:3) = eyeDat(:,2:3)*Exp.S.pixPerDeg;

% convert time to ephys units
eyeDat(:,1) = Exp.vpx2ephys(eyeDat(:,1));
[~, ~,id] = histcounts(frameTimes, eyeDat(:,1));
eyeAtFrame = eyeDat(id,2:3);




xpos = xpos - eyeAtFrame(:,1);
ypos = -ypos + eyeAtFrame(:,2);

figure(2); clf
plot(xpos, ypos, '.')

valid = hypot(eyeAtFrame(:,1), eyeAtFrame(:,2)) < 400;

% bin stimulus
ROI = [-200 -200 300 200]*2;
binSize = Exp.S.pixPerDeg;%/1.5;

xax = ROI(1):binSize:ROI(3);
yax = ROI(2):binSize:ROI(4);

[xx,yy] = meshgrid(xax, yax);

dims = [numel(yax) numel(xax)];
X = zeros(sum(valid), prod(dims));
for i = 1:NX
    disp(i)
    X = X + double(hypot(xpos(valid,i) - xx(:)', ypos(valid,i) - yy(:)') < binSize);
end

figure(3); clf
imagesc(X)


%
nlags = 12;


% ybar = mean(y,2);
% for cc = 1:NC
%     y(:,cc) = ybar;
% end

for ilag = 1:nlags
    lag = ilag;
    fprintf('Lag %d/%d\n', ilag, nlags)
    
    sta = (X(1:end-lag+1,:))'*y(lag:end, :);
    stas(:,ilag,:) = sta ./ sum(X)';
    
end


%
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

figure(1); clf
for kNeuron = 1:NC
    fprintf('Unit %d/%d\n', kNeuron, NC)
%     y = ytmp(:,kNeuron)-mean(ytmp(:,kNeuron)); % subtract mean to account for DC
%     sta = simpleRevcorr(X, y, nlags);
    sta = stas(:,:,kNeuron)';
    subplot(sx, sy, kNeuron, 'align')
    plot(sta)

    [u,s,v] = svd(sta);
    sd = sign(sum(u(:,1)));
    rfs(kNeuron).separability = s(1) / sum(diag(s));
    rfs(kNeuron).sta = sta;
    rfs(kNeuron).trf = u(:,1)*sd;
    rfs(kNeuron).srf = reshape(v(:,1)*sd, dims);
    
end


figure(2); clf


for kNeuron = 1:NC
    subplot(sx, sy, kNeuron, 'align')
    imagesc(xax/Exp.S.pixPerDeg, yax/Exp.S.pixPerDeg, rfs(kNeuron).srf)
    drawnow
end


%%







