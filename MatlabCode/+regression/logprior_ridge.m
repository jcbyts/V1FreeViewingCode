function [p,dp,H,logdetrm] = logprior_ridge(prvec,rho,iirdge,rhoNull)
% [p,dp,negCinv,logdetrm] = logprior_ridge(prvec,rho,iirdge,rhoNull)
%
% Evaluate a Gaussian log-prior at parameter vector prvec.
%
% Inputs:
%   prvec [n x 1] - parameter vector (last element can be DC)
%     rho [1 x 1] - ridge parameter (precision)
%  iirdge [v x 1] - indices to apply ridge prior to
% rhoNull [1 x 1] - prior precision for other elements
%
% Outputs:
%         p [1 x 1] - log-prior
%        dp [n x 1] - grad
%         H [n x n] - Hessian (negative inverse covariance matrix)
%    logdet [1 x 1] - log-determinant of Cinv (optional)


[nx,nvecs] = size(prvec);

if (nargin < 3) || isempty(iirdge)
    iirdge = (1:nx)';
    rhoNull = 1;
elseif (nargin < 4) || isempty(rhoNull)
    rhoNull = 1;
end

% Build diagonal inverse cov
Cinvdiag = rhoNull*ones(nx,1);
Cinvdiag(iirdge) = rho;

% Compute log-prior and gradient
if (nvecs == 1);
    dp = -prvec.*Cinvdiag; % grad
    p = .5*dp'*prvec; % logli
else
    % If multiple 'prvec' vectors passed in
    dp = -bsxfun(@times,prvec,Cinvdiag); % grad vectors
    p = .5*sum(bsxfun(@times,dp,prvec),1); % logli values
end

% Compute Hessian, if desired
if nargout > 2
    H = spdiags(-Cinvdiag,0,nx,nx);
    logdetrm = sum(log(Cinvdiag));
end
