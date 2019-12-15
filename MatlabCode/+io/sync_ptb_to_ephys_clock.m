function ptb2Ephys = sync_ptb_to_ephys_clock(Exp, ephysTrials)
% ptb2Ephys = syncPtb2EphysClock(Exp)
% Returns a single function handle for converting Psychtoolbox times into
% the ephys clock times for MarmoV

if nargin < 2
    % Get list of trials with electrophysiolgy timestamps
    ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS), Exp.D));
end

ephysClock = cellfun(@(x) x.START_EPHYS, Exp.D(ephysTrials));
try
    ptbClock = cellfun(@(x) x.STARTCLOCKTIME, Exp.D(ephysTrials));
catch
    ptbClock = cellfun(@(x) x.eyeData(1,6), Exp.D(ephysTrials));
end

ephysClock = [ephysClock; cellfun(@(x) x.END_EPHYS, Exp.D(ephysTrials))];
try
    ptbClock = [ptbClock; cellfun(@(x) x.ENDCLOCKTIME, Exp.D(ephysTrials))];
catch
    ptbClock = [ptbClock; cellfun(@(x) x.eyeData(end,6), Exp.D(ephysTrials))];
end

bad = isnan(ephysClock) | isnan(ptbClock);
ephysClock(bad) = [];
ptbClock(bad) = [];

% least-squares to synchronize
X = [ephysClock ones(numel(ptbClock), 1)];
w = (X'*X)\(X'*ptbClock);

% function to synchronize
ptb2Ephys = @(t) (t - w(2))/w(1);

fprintf('Synchronizing the Ephys and PTB clocks with %d valid strobes\n', numel(ephysClock))
totalErrorMs = sum((ephysClock - ptb2Ephys(ptbClock)).^2)*1e3;
fprintf('Total error (SSE): %02.5f ms\n', totalErrorMs)

% assert(totalErrorMs < .1, 'Clock sync failed')
