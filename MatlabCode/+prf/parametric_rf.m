function lambda = parametric_rf(params, kxy)
% parametric receptive field for hartley
% lambda = prf.parametric_rf(params, kxy)
% Inputs:
%   params [1 x 6] parameters of the model
%   kxy    [n x 2] number of points to evaluate (each row is kx , ky)
%
% Outpus:
%   lambda [n x 1] rate at each kx,ky point
%
%  parameters are:
%   1. Orientation Kappa
%   2. Orientation Preference
%   3. Spatial Frequency Preference
%   4. Spatial Frequency Sigma
%   5. Gain
%   6. Offset

% orientation = atan2(kxy(:,2),kxy(:,1));
orientation = kxy(:,1);
orientation(isnan(orientation)) = 0;

orientationTuning = -params(1) * cos(orientation - params(2)).^2;

% spatialFrequency = sqrt(sum(kxy.^2,2));
spatialFrequency = kxy(:,2);

spatialFrequencyTuning = (log(1 + spatialFrequency) - log(1 + params(3))).^2/2/params(4)^2;

lambda = params(5)*exp(orientationTuning - spatialFrequencyTuning) + params(6);
