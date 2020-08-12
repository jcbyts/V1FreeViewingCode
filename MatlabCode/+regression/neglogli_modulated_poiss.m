function [L,dL] = neglogli_modulated_poiss(prs,X,Y,nlsub,nlfun,dtbin,modulator)
% [L,dL,ddL] = neglogli_poiss(prs,X,Y,g,dtbin)
%
% Compute negative log-likelihood of data Y given rate g(X*prs).
%
% INPUT:
%       prs [M x 1] - parameters 
%         X [N x M] - design matrix
%         Y [N x 1] - observed Poisson random variables
%          nlfunsub - handle for subunit nonlinearity
%             nlfun - handle for transfer function
%     dtbin [1 x 1] - size of time bins of Y 
%
% OUTPUT: 
%         L [1 x 1] - negative log-likelihood
%        dL [M x 1] - gradient
%       ddL [M x M] - Hessian (2nd deriv matrix)
%
% written: 7 Apr 2012 (JW Pillow)


% Project parameters onto design matrix
z  = X*prs; % numerator

if nargin < 7
    modulator = 1;
end

etol = 1e-100;

if nargout==1
    % Compute neglogli
    
    g = nlsub(z);
    
    f = nlfun(g .* modulator);
    f = f*dtbin;
    f(f<etol)=etol;
    
    L = -Y'*log(f) + sum(f);
    L = double(gather(L)); % for gpu
    
elseif nargout == 2
    
    % Compute neglogli & Gradient
    [g, dg] = nlsub(z);
    [f, df] = nlfun(g .* modulator);
    
    % partial derivative wrt numerator
    df = df .* dg .* modulator .*dtbin;
    
    f = f.*dtbin; 
    f(f<etol)=etol;
    
    L = -Y'*log(f) + sum(f);
    
    % grad
    wts = df -( (Y.*df)./f);
%     wts = df -( (Y.*df)./f);
    dL = X'*wts;
    
    L = double(gather(L));
    dL = double(gather(dL));
    
elseif nargout == 3
    error('Hessian requested, but it has not been implemented. Turn HessObj off')
end
