function qf = qfsmooth3D(dims, lambda)
% qf = qfsmooth3D(dims, lambda)
    
    if nargin < 2
        lambda = 1;
    end
    
    numt = dims(1);
    numx = dims(2);
    numy = dims(3);
    
    if numel(lambda)==2
        lambda_space = lambda(2);
        lambda_time = lambda(1);
    else
        lambda_space = lambda;
        lambda_time = lambda;
    end
        
    Dt = qfsmooth1D(numt);
    Dxy = qfsmooth(numx, numy);
    qf = kron(lambda_space*Dxy,lambda_time*Dt);
end