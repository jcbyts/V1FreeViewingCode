function model = doRegression(XX, XY, YY, ntY, dspec)
% model = doRegression(dspec,XX,XY,YY,nTY)
%
% Perform linear regression on the binned spike train.
%   - Estimates linear kernel via ridge regression, with ridge parameter
%     set via empirical bayes ("evidence optimization").
%
% Input
%   XX [N x N] - autocovariance of regressors
%   XY [N x 1] - cross-covariance of regressors and response
%   YY [1 x 1] - response variance
%   ntY [1 x 1] - number of time points in response
%   dspec: carries info about design matrix
%
% Output
%   model.khat      = khat;
%   model.param.regressionMode
%   model.param.bilinearMode
%   model.param.bilinearRank
%   model.param.bilinearMethod
%   model.hyperprs  = hyperprs;
%   model.ntY       = ntY;
%   model.fnlin     = @(x) x;
%
% See also: loadData, compLinPrediction, decode
%
% Relevant options:
%    dspec.param.regressionMode: 'LS', 'ARD', 'SMOOTH ARD'
%    dspec.param.bilinearMode: 'ON', 'OFF' (low-rank coherence kernel)
%    dspec.param.bilinearRank: (integer)
%    dspec.param.bilinearCovariate: (string)
%           must match one of dspec.covar.desc
%
% Copyright 2010 Pillow Lab

if nargin==1 && isstruct(XX) && isfield(XX, 'XX')
    dspec=XX;
    XX=dspec.XX;
    YY=dspec.YY;
    XY=dspec.XY;
    ntY=dspec.ny;
end

if isstruct(XX) && ~isfield(XX, 'XX') % depreciated argument call with dspec in front
    tmp=XX;
    XX=XY;
    XY=YY;
    ntY=dspec;
    dspec=tmp;
end

if nargin==3
    X=XX;
    Y=XY;
    dspec=YY;
    XX=X'*X;
    XY=X'*Y;
    YY=Y'*Y;
    ntY=numel(Y);
    
    if ischar(dspec)
        tmp=dspec;
        dspec=struct;
        dspec.param.regressionMode=tmp;
        if strcmpi(tmp, 'SMOOTH ARD')
            dspec.covar.edim=size(X,2);
        end
    end
end

if nargin==2
    X=XX;
    Y=XY;
    XX=X'*X;
    XY=X'*Y;
    YY=Y'*Y;
    ntY=numel(Y);
end

if ~exist('dspec', 'var')
    dspec=struct();
    dspec.param.verbose=true;
    dspec.param.regressionMode='LS';
end

if ~isfield(dspec.param, 'verbose')
    dspec.param.verbose = false;
end

if ~isfield(dspec, 'edim')
    if isfield(dspec, 'covar')
        dspec.edim=sum([dspec.covar.edim]);
    else
        dspec.covar.edim=size(XX,2);
        dspec.edim=size(XX,2);
    end
end
% dspec fields
% .covar
%       .edim
%       .label
%       .desc
%
% .param
%       .regressionMode
%       .bilinearRank
%       .bilinearMode
%       .bilinearCovariate
% ---------------------------------------------------
% Do linear regression

% Optimization parameters
opts.maxiter = 50;   % maximum number of iterations
opts.tol = 1e-10;    % step size for determining convergence
opts.maxalpha = 1e5;
opts.maxrho = .999;
opts.verbose = false;

model=struct('khat', [], 'param', [], 'hyperprs', [], 'ntY', [], 'fnlin', []);
switch upper(dspec.param.regressionMode)
    case {'NONE', 'LS', 'ML', 'MLE'}
        model.param.regressionMode='NONE';
        khat = XX \ XY; % non-regularized least squares
        hyperprs.nsevar = (YY - 2*khat'*XY + khat'*XX*khat) / (ntY - 1);
        hyperprs.lambda = 0;
    case 'SMOOTH ARD'
        model.param.regressionMode='SMOOTH ARD';
        param = struct('xx',XX,'yy',YY','xy',XY,'ny',ntY);
        strtInds = cumsum([dspec.covar(1:end-1).edim])+1; % between each column
        strtInds = [strtInds dspec.edim]; % dc term
        
        [khat,hyperprs] = autoCorrRidgeRegress(param, strtInds, opts);
        hyperprs.lambda = hyperprs.alpha * hyperprs.nsevar;
    case 'RIDGE'
        model.param.regressionMode='RIDGE';
        % compute EB ridge regression estimate
        [khat,alpha,nsevar] = autoRidgeRegress(XX, XY, YY, ntY, opts);
        hyperprs = struct('alpha', alpha, 'nsevar', nsevar);
        hyperprs.lambda = hyperprs.alpha * hyperprs.nsevar;
    otherwise
        error('Unknown regression mode');
end

% ------------------------------------------------------
% Do bilinear regression for coherence kernel
if isfield(dspec.param, 'bilinearMode')
    covLabels = {dspec.covar(:).label};
    covDesc   = {dspec.covar(:).desc};
    
    switch upper(dspec.param.bilinearMode)
        case 'ON'
            brank = dspec.param.bilinearRank;
            
            % identify which coefficients are related to stimulus kernel
            [~,id] = findFile(covDesc, dspec.param.bilinearCovariate);
            grpCovLabels = covLabels(id);
            dspec.param.bilinearGroupList = id;
            nbil = length(dspec.param.bilinearGroupList); % number of coherence filters
            nprsperf = [dspec.covar(id).edim];
            assert(all(nprsperf==nprsperf(1)), 'all filters in bilinear group must be the same size')
            nprsperf = nprsperf(end); % # of params per coh filter
            ncohprs = nbil*nprsperf; %#ok<*NASGU>
            iicoh = cell2mat(buildGLM.getDesignMatrixColIndices(dspec, grpCovLabels));
            kcohdims = [nprsperf,nbil];  % size of coherence-related filter kcoh
            
            % Do bilinear optimization (coordinate ascent)
            if ~dspec.param.verbose; bl_opts.Display = 'none'; end
            khat = bilinearMixRegress_coordAscent(XX, XY, kcohdims, brank, ...
                iicoh, hyperprs.lambda, bl_opts);
        case 'OFF'
        otherwise
            error('Unknown bilinear mode');
    end
end

% output
model.khat      = khat;
model.hyperprs  = hyperprs;
model.ntY       = ntY;
model.fnlin     = @(x) x;