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
[bh,ah] = butter(3, (500/Fs)*2, 'high');

% lowpass filter
[b,a] = butter(3, (200/Fs)*2, 'low');

info = io.loadEphysInfo(ops);

NEW_FS = 1000; % HARDCODED -- downsample to 1kHz

downStep = ceil(Fs/NEW_FS);
nNewSamp = ceil(nTotSamp/downStep);

data = zeros(nNewSamp, Nchan);

ctr = 1;
padding = 1e3;
batchStarts = (0:(nBatch-1))*batchSize*Nchan*2;
fprintf('Extracting Spike Band Power for %d batches\n', nBatch)
for iBatch = 1:nBatch
    fprintf('Batch %d/%d\n', iBatch, nBatch)
    if iBatch > 1      
        fseek(fid, batchStarts(iBatch)-padding*Nchan*2, 'bof');
    end
    if iBatch == nBatch
        bufferSize = [Nchan batchSize + mod(nTotSamp, batchSize)];
        validIdx = 1:size(bufferSize,2);
    elseif iBatch == 1
        bufferSize = [Nchan batchSize + padding];
        validIdx = 1:batchSize;
    else
        bufferSize = [Nchan batchSize + 2*padding];
        validIdx = padding + (1:batchSize);
    end
    
    % read data
    data_ = double(fread(fid, bufferSize, '*int16'));
    
    data_ = data_ - mean(data_);
    
    % highpass
    data_ = filtfilt(bh,ah,data_');
    
    if plotit
        figure(111); clf
        subplot(211, 'align')
        plot(bsxfun(@plus, data_(validIdx(1) + (1:Fs),:)/500, (1:Nchan)), 'k'); hold on
    end
    
    % lowpass
    data_ = filtfilt(b,a,abs(data_));
    
    % remove padding
    data_ = data_(validIdx,:);
    
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
    
    if plotit
        arg = input('continue plotting?');
        if arg==0
            plotit = false;
        end
    end
        
    
    try
        ctr = ix(end);
    catch
        break
    end
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

[timestamps, ia] = unique(timestamps);
data = data(ia,:);

% remove artifacts
data = preprocess.removeChannelArtifacts(data, 150, 12, 50, true);

fprintf('Saving meta data...\t')
save(fMUA, '-v7.3', 'data', 'timestamps', 'xcoords', 'ycoords', 'zcoords')
% fprintf('[%02.2fs]\n', toc)

