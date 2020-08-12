function model = doRegressionPoisson(X, Y, dspec, ndx, dt, rho, colInds)
% Helper for poisson regression fits
% model = doRegressionPoisson(X, Y, dspec, ndx, dt, rho, colInds)
% INPUTS:
%   X [nsamples x ndims] Stimulus
%   Y [nsamples x 1]     Count
%   dspec [struct] or [neuroGLM object]
%       .covar - array of covariates
%           .label
%           .edim
%           .desc
%       .edim  - total dimensionality
%       .unitoftime
%       .binSize
%       .model
%           .regressionMode
%           .bilinearMode
%           .bilinearRank
%           .bilinearCovariate

import regression.*

model=[];

k0=(X'*X + eye(size(X,2)))\(X'*Y);

assert(isfield(dspec, 'model')|isprop(dspec, 'model'), 'model is a required field of dspec')
if ~isfield(dspec.model, 'regressionMode')
    dspec.model.regressionMode='ML';
end

if ~exist('ndx', 'var') || isempty(ndx)
    ndx = 1:numel(Y);
end

if ~exist('dt', 'var')
    dt = 1;
end

if ~exist('rho', 'var')
    rho=.2;
end

if isfield(dspec.model, 'optimOpts') && isa(dspec.model.optimOpts, optim.options.Fminunc)
    optimOpts=dspec.model.optimOpts;
else
    optimOpts = optimoptions(@fminunc, 'Display', 'iter', 'Algorithm','trust-region',...
        'GradObj','on','Hessian','on');
%     optimOpts = optimset('LargeScale', 'on', 'GradObj', 'on', 'Hessian', 'on', 'Display', 'iter');
end

if isfield(dspec.model, 'nlfun') && isa(dspec.model.nlfun, 'function_handle')
    nlfun=dspec.model.nlfun;
else
    nlfun = @expfun; % canonical link @logexp1;
end

switch dspec.model.regressionMode
    case {'MLEXP', 'MLSR', 'MLE', 'ML'}
        lfun = @(w) neglogli_poiss(w, X(ndx,:), Y(ndx), nlfun, dt);
        [wmle, fval , ~, ~, ~, H] = fminunc(lfun, k0, optimOpts);
        model.khat      = wmle;
        model.fnlin     = nlfun;
        model.SDebars   = sqrt(diag(inv(H)));
        model.fval=fval;
        model.dt=dt;
        model.df=numel(wmle);
    case {'Ridge', 'RIDGE'}
        [wRidge,rho,SDebars,~,logevid] = autoRegress_PoissonRidge(X(ndx,:),Y(ndx),nlfun,1:(size(X,2)-1),.1,[.1 1 10],k0);
        model.khat      = wRidge;
        model.fnlin     = nlfun;
        model.SDebars   = SDebars;
        model.rho       = rho;
        model.fval=logevid;
        model.dt=1;
        model.df=numel(wRidge);
    case {'RidgeFixed', 'RIDGEFIXED'}
        mstruct.neglogli = @neglogli_poiss;
        mstruct.logprior = @logprior_ridge;
        mstruct.liargs = {X(ndx,:),Y(ndx),nlfun,dt};
        mstruct.priargs = {1:(size(X,2)-1),.1};
        lfpost = @(w)(neglogpost_GLM(w,rho,mstruct));
        [wRidge, fval, ~, ~, ~, H] = fminunc(lfpost, k0, optimOpts);
        model.khat      = wRidge;
        model.fnlin     = nlfun;
        model.SDebars   = sqrt(diag(inv(H)));
        model.rho       = rho;
        model.fval=fval;
        model.dt=dt;
        model.df=numel(wRidge);
        
    case {'RidgeGroup'}
        mstruct.neglogli = @neglogli_poiss;
        mstruct.logprior = @logprior_ridge;
        assert(exist('colInds', 'var')>0, 'you must specify the indices of the covariates you want to penalize with L2')
        mstruct.liargs = {X(ndx,:),Y(ndx),nlfun,dt};
        mstruct.priargs = {colInds,.1};
        lfpost = @(w)(neglogpost_GLM(w,rho,mstruct));
        [wRidge, fval, ~, ~, ~, H] = fminunc(lfpost, k0, optimOpts);
        model.khat      = wRidge;
        model.fnlin     = nlfun;
        model.SDebars   = sqrt(diag(inv(H)));
        model.rho       = rho;
        model.fval=fval;
        model.dt=dt;
        model.df=numel(wRidge);
        
    case {'RidgeGroupCV'}
        mstruct.neglogli = @regression.neglogli_poiss;
        mstruct.logprior = @logprior_ridge;
        assert(exist('colInds', 'var')>0, 'you must specify the indices of the covariates you want to penalize with L2')
        mstruct.liargs = {X(ndx,:),Y(ndx),nlfun,dt};
        mstruct.priargs = {colInds,.1};
        nRho=numel(rho);
        if nRho>1
            [hprsMax,wmapMax] = gridsearch_GLMevidence(k0,mstruct,rho);
            fprintf('best grid point: rho (precision)=%.1f\n', hprsMax);
        end

        [fval, ~, H]=neglogpost_GLM(wmapMax, hprsMax, mstruct);
        model.khat      = wmapMax;
        model.fnlin     = nlfun;
        model.SDebars   = sqrt(diag(inv(H)));
        model.rho       = hprsMax;
        model.fval=fval;
        model.dt=dt;
        model.df=numel(wmapMax);
end

if isfield(dspec.model, 'bilinearMode')
    covLabels = {dspec.covar(:).label};
    covDesc   = {dspec.covar(:).desc};
    switch upper(dspec.model.bilinearMode)
        case 'ON'
            brank = dspec.model.bilinearRank;
            
            % identify which coefficients are related to stimulus kernel
            [~,id] = findFile(covDesc, dspec.model.bilinearCovariate);
            grpCovLabels = covLabels(id);
            dspec.model.bilinearGroupList = id;
            nbil = length(dspec.model.bilinearGroupList); % number of coherence filters
            nprsperf = [dspec.covar(id).edim];
            assert(all(nprsperf==nprsperf(1)), 'all filters in bilinear group must be the same size')
            nprsperf = nprsperf(end); % # of params per coh filter
            ncohprs = nbil*nprsperf;
            iicoh = cell2mat(getDesignMatrixColIndices(dspec, grpCovLabels));
            kcohdims = [nprsperf,nbil];  % size of coherence-related filter kcoh
            
            % Do bilinear optimization (coordinate ascent)
            [khat, SDebars, ~, ~,~,fval] = bilinearMixRegress_Poisson(X(ndx,:), Y(ndx), kcohdims, brank, ...
                iicoh, nlfun, dt, [], model.khat);
            
            model.khat      = khat;
            model.fnlin     = nlfun;
            model.SDebars   = SDebars;
            model.fval=fval;
            model.dt=dt;
            model.df=numel(khat)-ncohprs+nbil+nprsperf;
            
            return
        case 'OFF'
        otherwise
            error('Unknown bilinear mode');
    end
end

%% get Design matrix column indices
function [idx] = getDesignMatrixColIndices(dspec, covarLabels)
% Input
%   dpsec: design specification structure
%   covarLabels: 'str' or {'str'} - label(s) of the covariates
% Outut
%   idx: {} - column indices of the design matrix that correspond to the
%	    specified covariates

subIdxs = getGroupIndicesFromDesignSpec(dspec);

if ~iscell(covarLabels)
    covarLabels = {covarLabels};
end

labels={dspec.covar.label}';
labels=[labels num2cell((1:numel(labels))')]';
idxmap=struct(labels{:});
idx = cell(numel(covarLabels), 1);

for k = 1:numel(covarLabels)
    idx{k} = subIdxs{idxmap.(covarLabels{k})}(:);
end



%% get group indices
function subIdxs = getGroupIndicesFromDesignSpec(dspec)
% Cell of column indices that corresponds to each covariate in the design matrix
% subIdxs = getGroupIndicesFromDesignSpec(dspec)

subIdxs = {};
k = 0;

for kCov = 1:numel(dspec.covar)
    edim = dspec.covar(kCov).edim;
    subIdxs{kCov} = k + (1:edim); %#ok<AGROW>
    k = k + edim;
end









