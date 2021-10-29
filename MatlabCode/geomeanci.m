function [gm, ci_95] = geomeanci(x)
% calculate the geometric mean and 95% confidence intervals using logs
% [gm, ci] = geomeanci(x)
% Inputs:
%   x [n x 1] vector of values
% Outputs:
%   gm [1 x 1] geometric mean of x
%   ci [1 x 2] lower and upper 95% confidence interval

% to calculate the geometric mean and standard error of the
% geometric mean, compute the mean and se of the log of the ratio,
% then exponentiate
logx = log(x);
n = numel(x);

log_gm  = mean(logx); % log geometric mean
log_se = std(logx)/sqrt(n); % standard error (log of geometric mean)
log_ci_95 = [log_gm-2*log_se log_gm+2*log_se]; % 95% confidence intervals (in log space)

% exponentiate to get back to the geomean / se
gm = exp(log_gm);
ci_95 = exp(log_ci_95);