function [w_hat,SDebars, wt,wx,wlin,fval] = bilinearMixRegress_Poisson(X,Y,wDims,p,indsbilin,nlfun, dtbin, opts, w0)
% [w_hat,SDebars, wt,wx,wlin,fval] = bilinearMixRegress_Poisson(xx,xy,wDims,p,indsbilin,nlfun, dtbin, opts)
% or 
% [w_hat,SDebars, wt,wx,wlin] = bilinearMixRegress_Poisson(xx,xy,wDims,p,indsbilin,mstruct, opts)
% 
% Computes regression estimate with a bilinear parametrization of part of
% the parameter vector.
%
% Finds solution to argmin_w y'*log(x'*w) - sum(x'*w)
% where part of w is parametrized as vec(wt*wx')
%
% Inputs:
%   X  - [ns x nx] design matrix
%   Y  - spike count
%   wDims - [nt, nx] two-vector specifying # rows & # cols of bilinearly parametrized w
%   p - rank of bilinear filter
%   indsbilin - indices to be parametrized bilinearly (the rest parametrize linearly)
%   nlfun - nonlinearity
%   dtbin - delta time for poisson likelihood
%   opts - options struct (optional)
%          fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
%
%     mstruct.neglogli=@neglogli_poiss;
%     mstruct.logprior=[];
%     mstruct.nlfun=g;
%     mstruct.dtbin=dtbin;
%     mstruct.liargs={mstruct.nlfun,mstruct.dtbin};
%     mstruct.hprs=nan;
% Outputs:1
%   w  = estimate of full param vector
%   wlin = linearly parametrized portion
%   wt = column vectors (bilinear portion)
%   wx = row vectors (bilinear portion)
%   fval=likelihood value at end
%
% $Id$

import regression.*

if ~exist('dtbin', 'var')
    dtbin = 1;
end

if ~exist('nlfun', 'var')
    nlfun = @expfun;
end

if isstruct(nlfun)
	mstruct=nlfun;
    mstruct.liargs={mstruct.nlfun, mstruct.dtbin};
else
    mstruct.neglogli=@neglogli_poiss;
    mstruct.logprior=[];
    mstruct.nlfun=nlfun;
    mstruct.dtbin=dtbin;
    mstruct.liargs={mstruct.nlfun,mstruct.dtbin};
    mstruct.hprs=nan;
end

postfun = @(w,X,Y) posterior(w,X,Y, mstruct);
    


if (nargin < 9) || isempty(opts)
    opts.default = true;
end

if ~isfield(opts, 'MaxIter'); opts.MaxIter = 25; end
if ~isfield(opts, 'TolFun');  opts.TolFun = 1e-6; end
if ~isfield(opts, 'Display'); opts.Display = 'iter'; end

% Set some params
nw = size(X,2);
nbi = length(indsbilin);
nlin = nw-nbi;
nt = wDims(1);
nx = wDims(2);
nwt = p*nt;
nwx = p*nx;
It = speye(nt);
Ix = speye(nx);
Ilin = speye(nlin);

% Permute indices of XX and XY so that linear portion at beginning
indslin = setdiff(1:nw,indsbilin);  % linear coefficient indices
indsPrm = [indsbilin(:); indslin(:)]; % re-ordered indices
X = X(:,indsPrm);

iibi = 1:nbi;     % new bilinear indices (first ones)
iilin = nbi+1:nw; % new linear indices (last ones)

xx = X'*X;
xy = X'*Y;
% Initialize estimate of w by linear regression and SVD
if ~exist('w0', 'var')
    w0 = (xx+eye(size(xx)))\xy;
end
wlin = w0(iilin);
[wt,s,wx] = svd(reshape(w0(iibi),nt,nx));
wt = wt(:,1:p)*sqrt(s(1:p,1:p));
wx = sqrt(s(1:p,1:p))*wx(:,1:p)';

% Start coordinate ascent
w = [vec(wt*wx); wlin];
fvalfun = @(w) postfun(w, X, Y);
% fvalfun = @(w) neglogli_poiss(w, X, Y, g, dtbin);
fval = fvalfun(w);

fchange = inf;
iter = 1;
if strcmp(opts.Display, 'iter')
    fprintf('--- Coordinate descent of bilinear loss ---\n');
    fprintf('Iter 0: fval = %.4f\n',fval);
end

optimOpts = optimoptions('fminunc', 'algorithm', 'trust-region', 'PrecondBandWidth', 1, 'GradObj', 'on', 'Hessian', 'on');
while (iter <= opts.MaxIter) && (fchange > opts.TolFun)
    
    % Update temporal components
    wx=bsxfun(@rdivide, wx, sqrt(sum(wx.^2)));
    Mx = blkdiag(kron(wx',It),Ilin);
    Xproj = X*Mx;
%     C=Mx'*xx*Mx;
%     w0 = (C+eye(size(C)))\(Mx'*xy);
    lfun = @(w) postfun(w, Xproj, Y);
%     lfun = @(w) neglogli_poiss(w, Xproj, Y, g, dtbin);
%     wt = fminunc(lfun, w0, optimOpts);
    w0=[wt(:); wlin(:)];
    wt = fminunc(lfun, w0, optimOpts);
    wlin=wt(nwt+1:end);
    wt = reshape(wt(1:nwt), nt,p);
    
%     wt=bsxfun(@rdivide, wt, sqrt(sum(wt.^2)));
    
    % Update spatial components
    
    Mt = blkdiag(kron(Ix, wt),Ilin);
%     C=Mt'*xx*Mt;
%     wx0 = (C+eye(size(C)))\(Mt'*xy);
    Xproj = X*Mt;
    lfun = @(w) postfun(w, Xproj, Y);
%     lfun = @(w) neglogli_poiss(w, Xproj, Y, g, dtbin);
%     wx = fminunc(lfun, wx0, optimOpts);
    wx = fminunc(lfun, [wx(:); wlin(:)], optimOpts);
    
    wlin = wx(nwx+1:end);
    wx = reshape(wx(1:nwx),p,nx);

    % Compute size of change 
    w = [vec(wt*wx);wlin];
    fvalnew = fvalfun(w);
    fchange = fval-fvalnew;
    fval = fvalnew;
    iter = iter+1;
    if strcmp(opts.Display, 'iter')
	fprintf('Iter %d: fval = %.4f,  fchange = %.4f\n',iter-1,fval,fchange);
    end
end

[fval, ~, H] = fvalfun(w);
SDebars = sqrt(diag(inv(H)));
% finally, put indices of w back into correct order
w_hat = zeros(nw,1);
w_hat(indsbilin) = vec(wt*wx);
w_hat(indslin) = wlin;

function [L, dL, ddL] = posterior(w,X,Y,mstruct)
    if isempty(mstruct)||isempty(mstruct.logprior)
        liargs={X,Y,mstruct.liargs{:}}; %#ok<*CCAT>
        [L, dL, ddL] = mstruct.neglogli(w,liargs{:});
    else
        mstruct.liargs={X,Y,mstruct.liargs{:}}; %#ok<*CCAT>
        mstruct.priargs={1:size(X,2)-1, .1};
        [L, dL, ddL] = neglogpost_GLM(w,mstruct.hyperprs, mstruct);
    end
    

    
