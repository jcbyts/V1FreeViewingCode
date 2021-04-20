function Zr = radialavg(I, xax, nbins, win)

if nargin < 4
    win = 1;
end

sz = size(I);

% yax = -floor(sz(1)/2):ceil(sz(1)/2);
% xax = -floor(sz(2)/2):ceil(sz(2)/2);
yax = xax;

[xx,yy] = meshgrid(xax, yax);

r = hypot(xx,yy);

if numel(nbins)>1
    rbins = nbins;
    nbins = numel(rbins);
else
    rbins = linspace(min(r(:)), max(r(:)), nbins+1);
end

Zr = zeros(1,nbins); % vector for radial average

% nans = ~isnan(I); % identify NaNs in input data
% loop over the bins, except the final (r=1) position
for j=1:nbins-1
	% find all matrix locations whose radial distance is in the jth bin
	bins = r>=rbins(max(1, j-win)) & r<rbins(min(nbins, j+1+win));
	
	% exclude data that is NaN
% 	bins = logical(bins .* nans);
	
	% count the number of those locations
	n = sum(bins(:));
	if n~=0
		% average the values at those binned locations
        w = I.*bins;
 		Zr(j) = sum(w(:))/n;
	else
		% special case for no bins (divide-by-zero)
		Zr(j) = NaN;
	end
end




