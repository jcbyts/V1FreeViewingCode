function [L,dL] = neglogli_poiss_subunit(prs,X,Y,nlsub,nlfun,dtbin)
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
nd = size(X,2);
iinum = 1:nd;
iiden = iinum + nd;

znum = X*prs(iinum); % numerator
zden  = X*prs(iiden); % gain dirve

etol = 1e-100;

if nargout==1
    % Compute neglogli
    
    gnum = nlsub{1}(znum);
    gden = nlsub{2}(zden);
    
    f = nlfun(gnum ./ gden);
    f = f*dtbin;
    f(f<etol)=etol;
    
    L = -Y'*log(f) + sum(f);
    L = double(gather(L)); % for gpu
    
elseif nargout == 2
    
    % Compute neglogli & Gradient
    [gnum,dgnum] = nlsub{1}(znum);
    [gden,dgden] = nlsub{2}(zden);
    
    [f, df] = nlfun(gnum./gden);
    
    % partial derivative wrt numerator
    
    dfnum = df.*dgnum;
    dfden = df.*dgden;
    df = [dfnum; dfden];
    
    f = f.*dtbin; df = df.*dtbin;
    f(f<etol)=etol;
    
    L = -Y'*log(f) + sum(f);
    
    % grad
%     wts = (df-(Y.*df./f));
    wtsnum = (dfnum-(Y.*dfnum./f));
    wtsden = (dfden-(Y.*dfden./f));
    dL = [X'*wtsnum; X'*wtsden];
    
    L = double(gather(L));
    dL = double(gather(dL));
    
elseif nargout == 3
    error('Not implemented yet')
end
