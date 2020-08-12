function [f,df,ddf]=logistic(x)
% e^x/(1+e^x)

ex=exp(x);
oneplus=1+ex;

f=ex ./ oneplus;

if nargout>1
    df=ex ./ oneplus.^2;
    if nargout >2
        ddf=-ex.*(ex-1) ./ oneplus.^3;
    end    
end

