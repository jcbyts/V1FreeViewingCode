function [g,mus] = vm_basis_deg(m, kD, includeNormalization)
% [g,mus] = vm_basis_deg(m, kD, includeNormalization)

if nargin < 3
    includeNormalization = true;
end

bs = 360/m;
mus = 0:bs:(360 - bs);

% myfun = @(k) exp(k*cosd(bs/2))/exp(k) - .5;
% kD = fsolve(myfun, 10);

if nargin < 2 || isempty(kD)
    kD = log(.5)/(cosd(bs/2)-1);
end

% kD = hw2k(bs/2);
b0basis = besseli(0, kD);

if includeNormalization
    g=@(theta) exp(kD*cosd(bsxfun(@minus, theta(:), mus)))/(360*b0basis);
else
    g=@(theta) exp(kD*cosd(bsxfun(@minus, theta(:), mus)));
end