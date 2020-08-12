function [qf] = qfsmooth1D2nd(numx)
    %[qf] = qfsmooth1D(numx)
    %Create a quadratic form for smoothness regularization based on
    %second-order derivative operator, for a one-dimensional signal
    D = zeros(numx+1,numx);
%     D(1, 1:2) = [2 -1];
    for ii = 2:numx-1
        D(ii,ii-1:ii+1) = [-1 2 -1];
    end
    
    qf = D'*D;
end