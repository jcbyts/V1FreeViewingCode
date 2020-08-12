function [p,dp,Cinv] = logprior_Cinv(prvec,lambda,Cinv)
% [p,dp,Cinv,logdetrm] = logprior_Cinv(prvec, lambda, Cinv)
%
% Evaluate Tikanov prior. Pass in a prior inverse covariance, get back the
% a gaussian prior evaluated for th


% % Add prior indep prior variance on DC coeff
% if DCflag
%     Cinv = blkdiag(Cinv,-rhoDC);
% end
Cinv = -Cinv * lambda;

dp = Cinv*prvec; % gradient
p = .5*sum(prvec.*dp,1); % log-prior
