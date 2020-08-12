function [w_hat,SDebars, wt,wx,wlin] = bilinearMixRegress_Bernoulli(X,Y,wDims,p,indsbilin, opts, tryEvidenceOptimization)
% [w_hat,SDebars, wt,wx,wlin] = bilinearMixRegress_Bernoulli(xx,xy,wDims,p,indsbilin,opts)
% 
% Computes regression estimate with a bilinear parametrization of part of
% the parameter vector.
%
% Finds solution to argmin_w w'x - log(1+exp(w'x))
% where part of w is parametrized as vec(wt*wx')
%
% Inputs:
%   X  - [ns x nx] design matrix
%   Y  - spike count
%   wDims - [nt, nx] two-vector specifying # rows & # cols of bilinearly parametrized w
%   p - rank of bilinear filter
%   indsbilin - indices to be parametrized bilinearly (the rest parametrize linearly)
%   opts - options struct (optional)
%          fields: 'MaxIter' [25], 'TolFun' [1e-6], 'Display' ['iter'|'off']
%
% Outputs:1
%   w  = estimate of full param vector
%   wlin = linearly parametrized portion
%   wt = column vectors (bilinear portion)
%   wx = row vectors (bilinear portion)
%

if (nargin < 8) || isempty(opts)
    opts.default = true;
end

if ~isfield(opts, 'MaxIter'); opts.MaxIter = 25; end
if ~isfield(opts, 'TolFun'); opts.TolFun = 1e-6; end
if ~isfield(opts, 'Display'); opts.Display = 'iter'; end
if ~exist('tryEvidenceOptimizaiton', 'var')
    tryEvidenceOptimization=false;
end

% Set some params
nw      = size(X,2);
nbi     = length(indsbilin);
nlin    = nw-nbi;
nt      = wDims(1);
nx      = wDims(2);
nwt     = p*nt;
nwx     = p*nx;
It      = speye(nt);
Ix      = speye(nx);
Ilin    = speye(nlin);

% Permute indices of XX and XY so that linear portion at beginning
indslin = setdiff(1:nw,indsbilin);  % linear coefficient indices
indsPrm = [indsbilin(:); indslin(:)]; % re-ordered indices
X = X(:,indsPrm);

iibi  = 1:nbi;     % new bilinear indices (first ones)
iilin = nbi+1:nw; % new linear indices (last ones)

xx = X'*X;
xy = X'*Y;
% Initialize estimate of w by linear regression and SVD
w0 = (xx+5*eye(size(xx)))\xy;
wlin = w0(iilin);
[wt,s,wx] = svd(reshape(w0(iibi),nt,nx));
wt = wt(:,1:p)*sqrt(s(1:p,1:p));
wx = sqrt(s(1:p,1:p))*wx(:,1:p)';

% Start coordinate ascent
w = [vec(wt*wx); wlin];
fvalfun = @(w) neglogli_bernoulliGLM(w, X, Y);
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
    Mx = blkdiag(kron(wx',It),Ilin);
    Xproj = X*Mx;
    w0 = (Mx'*xx*Mx)\(Mx'*xy);
    if tryEvidenceOptimization
        wt=autoRegress_logisticRidge(Xproj, Y, 1:size(Xproj,2), 10, [.1 .9 100], w0);
    else
        lfun = @(w) neglogli_bernoulliGLM(w, Xproj, Y);
        wt = fminunc(lfun, w0, optimOpts);
    end
    
    wt = reshape(wt(1:nwt), nt,p);
    
    % Update spatial components
    Mt = blkdiag(kron(Ix, wt),Ilin);
    wx0 = (Mt'*xx*Mt)\(Mt'*xy);
    Xproj = X*Mt;

    if tryEvidenceOptimization
        wx=autoRegress_logisticRidge(Xproj, Y, 1:size(Xproj,2), 10, [.1 .9 100], wx0);
    else
        lfun = @(w) neglogli_bernoulliGLM(w, Xproj, Y);
        wx = fminunc(lfun, wx0, optimOpts);
    end
    
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

% if tryEvidenceOptimizaiton
%     % Update temporal components
%     Mx = blkdiag(kron(wx',It),Ilin);
%     Xproj = X*Mx;
%     w0 = (Mx'*xx*Mx)\(Mx'*xy);
%     wt=autoRegress_logisticRidge(Xproj, Y, 1:size(Xproj,2), .5, [.1 .9 1], w0);
%     
%     wt = reshape(wt(1:nwt), nt,p);
%     
%     % Update spatial components
%     Mt = blkdiag(kron(Ix, wt),Ilin);
%     wx0 = (Mt'*xx*Mt)\(Mt'*xy);
%     Xproj = X*Mt;
%     wx=autoRegress_logisticRidge(Xproj, Y, 1:size(Xproj,2), .5, [.1 .9 1], wx0);
%     
%     wlin = wx(nwx+1:end);
%     wx = reshape(wx(1:nwx),p,nx);
% end

[~, ~, H] = fvalfun(w);
SDebars = sqrt(diag(inv(H)));
% finally, put indices of w back into correct order
w_hat = zeros(nw,1);
w_hat(indsbilin) = vec(wt*wx);
w_hat(indslin) = wlin;
