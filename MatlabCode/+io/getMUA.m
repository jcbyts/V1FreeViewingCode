function [data, timestamps, xcoords, ycoords, zcoords] = getMUA(ops, overwrite, plotit)

% fproc
if nargin < 3
    plotit = false;
end

if nargin < 2
    overwrite = false;
end


fMUA = fullfile(ops.root, 'MUA.mat');

if exist(fMUA, 'file') && ~overwrite
    load(fMUA)
    return
end

fid = fopen(ops.fbinary);

fseek(fid, 0, 'eof');
filesize = ftell(fid);

fseek(fid, 0, 'bof');

Nchan = ops.Nchan;

nTotSamp = filesize/Nchan/2;

nBatch = 1000;
batchSize = floor(nTotSamp/nBatch);

% lowpass filter
Fs = ops.fs;

% highpass 
[bh,ah] = butter(6, (500/Fs)*2, 'high');

% lowpass filter
[b,a] = butter(3, (200/Fs)*2, 'low');

info = io.loadEphysInfo(ops);

NEW_FS = 1000; % HARDCODED -- downsample to 1kHz

downStep = ceil(Fs/NEW_FS);
nNewSamp = ceil(nTotSamp/downStep);

data = zeros(nNewSamp, Nchan);

ctr = 1;
fprintf('Extracting Spike Band Power for %d batches\n', nBatch)
for iBatch = 1:nBatch
    fprintf('Batch %d/%d\n', iBatch, nBatch)
    if iBatch == nBatch
        bufferSize = [Nchan batchSize + mod(nTotSamp, batchSize)];
    else
        bufferSize = [Nchan batchSize];
    end
    
    % read data
    data_ = double(fread(fid, bufferSize, '*int16'));
    
    % highpass
    data_ = filter(bh,ah,data_');
    
    if plotit
        figure(111); clf
        subplot(211, 'align')
        plot(bsxfun(@plus, data_(1:Fs,:)/500, (1:Nchan)), 'k'); hold on
    end
    
    % lowpass
    data_ = filtfilt(b,a,abs(data_));
    
    if plotit
        plot(bsxfun(@plus, (data_(1:Fs,:))/100, (1:Nchan)), 'r');
    
    subplot(212, 'align')
    imagesc(data_(1:Fs,:)')
    axis xy
    
    xlabel('Sample')
        ylabel('Channel')
    end
    
    downdata = data_(1:downStep:end,:);
    ix = ctr + (1:size(downdata,1));
    
    % downsample
    data(ix,:) = downdata;
    
    ctr = ix(end);
   
end

% close up shop
fclose(fid);

% --- save output
newInfo = info;
newInfo.sampleRate = NEW_FS;
newInfo.fragments  = info.fragments(:)/downStep;
info = newInfo;

timestamps = io.convertSamplesToTime((1:size(data,1))', info.sampleRate, info.timestamps(:), info.fragments(:));

load(ops.chanMap);

fprintf('Saving meta data...\t')
save(fMUA, '-v7.3', 'data', 'timestamps', 'xcoords', 'ycoords', 'zcoords')
% fprintf('[%02.2fs]\n', toc)

