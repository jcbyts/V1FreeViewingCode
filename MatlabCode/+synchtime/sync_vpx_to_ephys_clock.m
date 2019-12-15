function vpx2Ephys = sync_vpx_to_ephys_clock(Exp, ephysTrials)
% vpx2Ephys = sync_vpx_to_ephys_clock(Exp, ephysTrials)
% Returns a single function handle for converting Eyetracker times into
% the ephys clock times for MarmoV

if nargin < 2
    % Get list of trials with electrophysiolgy timestamps
    ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS), Exp.D));
end

ephysClock = cellfun(@(x) x.START_EPHYS, Exp.D(ephysTrials));
try
    vpxClock = cellfun(@(x) x.START_VPX, Exp.D(ephysTrials));
catch
    vpxClock = cellfun(@(x) x.eyeData(1,6), Exp.D(ephysTrials));
end

ephysClock = [ephysClock; cellfun(@(x) x.END_EPHYS, Exp.D(ephysTrials))];
try
    vpxClock = [vpxClock; cellfun(@(x) x.END_VPX, Exp.D(ephysTrials))];
catch
    vpxClock = [vpxClock; cellfun(@(x) x.eyeData(end,6), Exp.D(ephysTrials))];
end

bad = isnan(ephysClock) | isnan(vpxClock);
ephysClock(bad) = [];
vpxClock(bad) = [];

% % least-squares to synchronize
% X = [ephysClock ones(numel(vpxClock), 1)];
% w = (X'*X)\(X'*vpxClock);
% 
% % function to synchronize
% vpx2Ephys = @(t) (t - w(2))/w(1);
vpx2Ephys = synchtime.align_clocks(vpxClock, ephysClock);

% fprintf('Synchronizing the Ephys and Eyetracker clocks with %d valid strobes\n', numel(ephysClock))
% totalErrorMs = sum((ephysClock - vpx2Ephys(vpxClock)).^2)*1e3;
% fprintf('Total error (SSE): %02.5f ms\n', totalErrorMs)

% assert(totalErrorMs < .1, 'Clock sync failed')
