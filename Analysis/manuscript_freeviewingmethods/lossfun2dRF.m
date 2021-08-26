function loss = lossfun2dRF(params, x, y, robs, tkern)
% loss = lossfun2dRF(params, x, y, robs, tkern)
if nargin < 5
    lambda = max(gauss2Drf(params, x, y), 0);
else
    lambda = max(gauss2Drf(params, x, y, tkern), 0);
end

loss = -robs'*log(lambda) + sum(lambda);


