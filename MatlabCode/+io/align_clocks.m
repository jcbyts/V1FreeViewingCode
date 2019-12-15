function syncFun = align_clocks(clock1, clock2)
% syncFun = align_clocks(clock1, clock2)

bad = isnan(clock1) | isnan(clock2);
clock1(bad) = [];
clock2(bad) = [];

% least-squares to synchronize
X = [clock2 ones(numel(clock1), 1)];
w = (X'*X)\(X'*clock1);

% function to synchronize
syncFun = @(t) (t - w(2))/w(1);

fprintf('Synchronizing Clock 1 and Clock 2 with %d timestamps\n', numel(clock2))
totalErrorMs = sum((clock2 - syncFun(clock1)).^2)*1e3;
fprintf('Total error (SSE): %02.5f ms\n', totalErrorMs)