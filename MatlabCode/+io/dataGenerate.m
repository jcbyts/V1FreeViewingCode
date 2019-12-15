function dataGenerate(Exp, S, varargin)
% simple spike-triggered analysis of neuron responses
% simpleSTAanalyses(Exp, S, varargin)
% 
% Inputs:
%   Exp [struct]: marmoview experiment struct
%   S   [struct]: options struct from dataFactory
%       Latency: latency of the monitor
%       rect:  analysis ROI (in pixels, centered on gaze)
% Outputs:
%   several figures that are saved to disk
%
% Optional Arguments:
%   'stimulus': {'Gabor', 'Grating', 'Dots', 'BackImage', 'All'}
%   'testmode': logical. only use 10 trials (test the code)
%   'fft': logical. RF in fourier domain (energy)
%   'pixelsquaring': logical. square pixels before computing RF?
%   'binsize': discretization of space (in pixels)
%   'eyesmooth': smoothing window for eye position (sgolay fitler)
%   't_downsample': temporal downsampling factor (1 = no downsampling)
%   's_downsample': spatial downsampling factor (1 = no downsampling)
%   'includeProbe': whether to include the probe in the reconstruction

% varargin = {'t_downsample', 2, 's_downsample', 2, 'eyesmooth', 7};

ip = inputParser();
ip.addParameter('stimulus', 'Gabor')
ip.addParameter('testmode', true)
ip.addParameter('fft', false)
ip.addParameter('pixelsquaring', false)
ip.addParameter('binsize', 1)
ip.addParameter('eyesmooth', 3)
ip.addParameter('t_downsample', 1)
ip.addParameter('s_downsample', 1)
ip.addParameter('includeProbe', true)
ip.parse(varargin{:})

%% manual adjustment to rect
if ~isfield(S, 'rect')
    error('simpleSTAanalyses: you must specify a stimulus rect for analysis')
end

% get stimulus ROI
rect = S.rect;

% get clusters to analyze
if isfield(S, 'cids')
    cluster_ids = S.cids;
else
    cluster_ids = Exp.osp.cids;
end

spikeBinSize = 1/Exp.S.frameRate; % bin at the frame rate (4 ms bins)

%% Select trials to analyze
stimulusSet = ip.Results.stimulus;
trialProtocols = cellfun(@(x) x.PR.name, Exp.D, 'uni', 0);

% --- find the trials that we want to analyze
ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS) | ~isnan(x.END_EPHYS), Exp.D));
if (numel(ephysTrials)/numel(Exp.D)) < 0.6
    disp('Something is wrong. Assuming all trials had ephys')
    ephysTrials = 1:numel(Exp.D);
end

switch stimulusSet
    case {'Grating'}
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==1, Exp.D(validTrials)));
        
    case {'Gabor'}
        % Forage trials
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==4, Exp.D(validTrials)));
        
    case {'Dots'}
        % Forage trials
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        % dot spatial noise trials
        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));

        dotSize = cellfun(@(x) x.P.dotSize, Exp.D(validTrials));
        validTrials = validTrials(dotSize==min(dotSize));
    case {'BackImage'}
        validTrials = intersect(find(strcmp(trialProtocols, 'BackImage')), ephysTrials);
        
    otherwise
        % use all valid conditions (BackImage or
        validTrials = find(strcmp(trialProtocols, 'BackImage') | strcmp(trialProtocols, 'ForageProceduralNoise'));
        validTrials = intersect(validTrials, ephysTrials);
end


numValidTrials = numel(validTrials);
fprintf('Found %d valid trials\n', numValidTrials)

if numValidTrials == 0
    return
end

% smooth the eye position with an sgolay filter
eyePos = Exp.vpx.smo(:,2:3);
if ip.Results.eyesmooth > 1 
    
    % smoothing window must be odd
    if mod(ip.Results.eyesmooth-1,2)~=0
        smwin = ip.Results.eyesmooth - 1;
    else
        smwin = ip.Results.eyesmooth;
    end
    
    eyePos(:,1) = sgolayfilt(eyePos(:,1), 1, smwin);
    eyePos(:,2) = sgolayfilt(eyePos(:,2), 1, smwin);
    
end

% make sure we have the latency of the monitor / graphics card included
if ~isfield(S, 'Latency')
    Latency = 8.3;
else
    Latency = S.Latency;
end

if ip.Results.testmode
    fprintf('**** TEST MODE ****\nOnly using 10 trials\n')
%     validTrials = randsample(validTrials, 20);
    validTrials = validTrials(1:20);
end

binSize = ip.Results.binsize; % pixel
fprintf('Regenerating stimuli...\n')
[Stim, frameInfo] = regenerateStimulus(Exp, validTrials, rect, 'spatialBinSize', binSize, 'Latency', Latency, 'eyePos', eyePos, 'includeProbe', ip.Results.includeProbe);

if ip.Results.fft
    win = hanning(size(Stim,1))*hanning(size(Stim,2))';
    
    for iFrame = 1:size(Stim,3)
        I = Stim(:,:,iFrame);
        
        % kill DC
        I = I - mean(I(:));
        
        % window
        I = I .* win;
        
        % Fourier energy
        Stim(:,:,iFrame) = abs(fftshift(fft2(I)));
    end
end

% spatial downsampling
if ip.Results.s_downsample > 1
    [NNX, NNY] = size(downsample_image(Stim(:,:,1), ip.Results.s_downsample));
    tmp = zeros(NNX,NNY, size(Stim,3));
    disp('Downsampling images.')
    for tt = 1:size(Stim,3)
        tmp(:,:,tt) = downsample_image(Stim(:,:,tt), ip.Results.s_downsample);
    end
    Stim = tmp;
    frameInfo.dims = [NNY NNX];
end

NTall = size(Stim,3);

%% Get indices for analysis 
valdata = zeros(NTall,1);

bad = ~frameInfo.seedGood;
fprintf('%d frames excluded because the seed failed\n', sum(bad))
goodIx = ~bad;

bad = frameInfo.frameTimesOe==0; 
fprintf('%d frames excluded because the frameTime was 0\n', sum(bad(goodIx)))
goodIx = goodIx & ~bad;

valdata(goodIx) = 1;

fprintf('%d good frames at stimulus resolution (%0.2f sec)\n', sum(valdata), sum(valdata)*spikeBinSize);


% get eye pos labels
labels = Exp.vpx.Labels(frameInfo.eyeAtFrame(:,4));

% get saccade times
saccades = labels==2;
sstart = find(diff(saccades)==1);
sstop = find(diff(saccades)==-1);
if saccades(1)
    sstart = [1; sstart];
end

if saccades(end)
    sstop = [sstop; numel(saccades)];
end
slist = [sstart sstop];

%% Bin spike times

Y = binNeuronSpikeTimesFast(Exp.osp, frameInfo.frameTimesOe, spikeBinSize);
Y = Y(:,cluster_ids);

% reshape the stimulus
xax = rect(1):binSize:(rect(3)-binSize);
yax = rect(2):binSize:(rect(4)-binSize);

if ip.Results.fft % get coordinates in fourier domain
    sFs = Exp.S.pixPerDeg/binSize;
    nx = numel(xax);
    ny = numel(yax);
    xax = fftshift([0:floor((nx-1)/2), -floor(nx/2):-1]/nx*sFs);  % coordinates for Fourier components (space x dimension)
    yax = fftshift([0:floor((ny-1)/2), -floor(ny/2):-1]/ny*sFs);  % coordinates for Fourier components (space x dimension)
end

X = double(reshape(Stim, prod(frameInfo.dims), NTall))';

probeDist = frameInfo.probeDistance;
%% Temporal downsample
if ip.Results.t_downsample > 1
	X = downsample_time(X, ip.Results.t_downsample) / ip.Results.t_downsample;
	Y = downsample_time(Y, ip.Results.t_downsample);
	valdata = downsample_time(valdata, ip.Results.t_downsample) / ip.Results.t_downsample;
	valdata(valdata < 1) = 0;
    labels = downsample_time(labels, ip.Results.t_downsample) / ip.Results.t_downsample;
    slist = ceil(slist / ip.Results.t_downsample);
    probeDist = downsample_time(probeDist, ip.Results.t_downsample) / ip.Results.t_downsample;
end
dt = spikeBinSize * ip.Results.t_downsample;
NX = size(Stim,1);

fprintf('Size of stimulus: [%d x %d]\n', size(X,1), size(X,2))
fprintf('Size of Robs: [%d x %d]\n', size(Y,1), size(Y,2))

stim = X;  Robs = full(Y); 
opts = ip.Results;

%% save

fname = sprintf('%s_%s.mat', strrep(Exp.FileTag, '.mat', ''), ip.Results.stimulus);
fprintf('saving output to [$s]\n', fname)
dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
save(fullfile(dataDir, fname), '-v7.3', 'stim', 'Robs', 'valdata', 'labels', 'xax', 'yax', 'dt', 'NX', 'slist', 'opts', 'probeDist');
fprintf('Done\n')

% if your're dugging, you can look at this code below to make sure things
% are lined up
return
%% saccade triggered average


[an, ~, widx] = eventTriggeredAverage(mean(Robs,2), slist(:,2), [-1 1]*ceil(.2/dt));
figure(1); clf
plot(widx * dt, an / dt)

%% sike triggered average stimulus for small number of neurons
figure(1); clf
ix = valdata == 1 & labels == 1 & probeDist > 50;
for k = 1:min(25, size(Robs,2))
    sta = simpleSTC((stim(ix,:)), Robs(ix,k), ceil(.1/dt) );

    subplot(5, 5, k, 'align')
    imagesc(sta)
end

%% sample neuron
k = 1%k + 1;
nk = ceil(.1/dt);
ix = valdata == 1 & labels == 1 & probeDist > 100;
sta = simpleSTC(stim(ix,:), Robs(ix,k), nk);
sta = (sta - min(sta(:))) / (max(sta(:)) - min(sta(:)));
sta = flipud(sta);
figure(ip.Results.eyesmooth); clf
for i = 1:nk
    subplot(1,nk,i,'align')
    imagesc(reshape(sta(i,:), fliplr(frameInfo.dims)), [0 1])
    title(sprintf('%02.1f', 1e3*dt*i))
end

