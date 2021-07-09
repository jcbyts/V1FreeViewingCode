function [g,mus] = vm_ori_basis_deg(m, kD, includeNormalization)
% [g,mus] = vm_basis_deg(m, kD, includeNormalization)

if nargin < 3
    includeNormalization = true;
end

bs = 180/m;
mus = 0:bs:(180 - bs);

if nargin < 2 || isempty(kD)
    kD = (log(.5)/(cosd(bs/2)-1))/2;
end

b0basis = besseli(0, kD);

if includeNormalization
    g=@(theta) exp(kD*cosd(bsxfun(@minus, theta(:), mus)).^2)/(180*b0basis);
else
    g=@(theta) exp(kD*cosd(bsxfun(@minus, theta(:), mus)).^2);
end