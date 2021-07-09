function L = neglogli_poissGLM(lambda,R)

L = -R'*log(lambda) + sum(lambda);