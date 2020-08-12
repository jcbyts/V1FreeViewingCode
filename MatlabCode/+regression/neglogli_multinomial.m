function [negL,dnegL,H] = neglogli_multinomialGLM(wts,X,Y)
% [negL,dnegL,H] = neglogli_bernoulliGLM(wts,X,Y)
%
% Compute negative log-likelihood, gradient, and Hessian of multinomial
% logistic regression model 
%
% Inputs:
% wts [mk x 1] - weights: constants and weights for each of k-1 classes
%   X [N x m] - regressors
%   Y [N x k] - output (binary indicators with '1' indicating one of
%                      first k-1 classes, or all-zeros for k'th class)
%
% Outputs:
%    negL - negative loglikelihood
%   dnegL - gradient
%       H - Hessian
%
% Details: 
% --------
% Describes mapping from vectors x to a discrete variable y taking values
% from one of k classes. Formally:
%     P(Y = j|x) = 1/Z exp(x*w_j), where
%              Z = \sum_i=1^k  exp(x*w_i)
% - No constant is included automatically, so X should include a
%   column of 1's to incorporate a constant.
% - The output Y should be represented as a binary matrix of size N x k,
%   with '1' indicating the class for each row.
% - Note the model is overparametrized since we don't need weights for k'th
%   class, making it more appropriate for MAP estimation (e.g., where
%   we have smoothness constraints on the weights).

npred = size(X,2); % number of predictors (dimensionality of input space)
nclass = size(Y,2); % number of classes
nw = npred*nclass; % total number of weights in model
w = reshape(wts,npred,nclass); 

% Compute projection of stimuli onto weights
xproj = X*w;

% for numerical stability (to avoid large exp(x)) subtract max from each row
xproj = bsxfun(@minus,xproj,max(xproj,[],2)); 

if nargout <= 1
    negL = -sum(sum(Y.*xproj)) + sum(logexpsum(xproj)); % neg log-likelihood
elseif nargout >= 2
    [f,df] = logexpsum(xproj); % evaluate log-normalizer & deriv
    
    negL = -sum(sum(Y.*xproj)) + sum(f); % neg log-likelihood
    dnegL = reshape(X'*(df-Y),[],1);                    % gradient

    if nargout > 2  
        % compute Hessian
        
        % Compute stimulus weighted by df for each class
        Xdf = reshape(bsxfun(@times,X,reshape(df,[],1,nclass)),[],nw);
        
        % Build center block-diagonal portion 
        H = zeros(nw);
        for jj = 1:nclass
            inds = (jj-1)*npred+1:jj*npred;
            H(inds,inds) = X'*Xdf(:,inds);
        end
        H = H-Xdf'*Xdf;
        %,reshape(df
        %H = X'*bsxfun(@times,X,ddf); % Hessian
        
 %       H = H+2*eye(nclass*npred);

    end

   % dnegL = dnegL+ 2*w(:);

end

% negL = negL + w(:)'*w(:);

% -------------------------------------------------------------
% ----- logexpsum Function (log-normalizer) --------------------
% -------------------------------------------------------------

function [f,df,ddf] = logexpsum(x)
%  [f,df] = logexpsum(x);
%
%  Computes:  f(x) = log(sum(exp(x))
%  and its 1st and 2nd derivatives, where sum is along rows of x
%
%  Returns: column vectors of 

f = log(sum(exp(x),2));

if nargout > 1
    df = bsxfun(@rdivide,exp(x),sum(exp(x),2));
end
