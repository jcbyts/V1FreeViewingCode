function v = vonmisesnest(x, params)
% nested von mises function that can return an orientation or
% direction-tuned curve
%
% INPUT:
% x - radians
% params - {mu, kappa, A, b, lambda}
%   mu - central tendency   
%   kappa - dispersion
%   A - amplitude
%   b - baseline
%   lambda - amount of direction

k = params(2);
mu = params(1);
A = params(3);
b = params(4);
if numel(params)==4
    lambda = 1;
else
    lambda = params(5);
end

v = b + A*exp(k*cos(x - mu) - k) + A*lambda*exp(k*cos(x - mu - pi) - k);
