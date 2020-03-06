%
% This M-file computes the temporal response of an M-type retinal ganglion
% cell as reported by Benardete and Kaplan (1999)
% Visual Neuroscience (16):355-368
%
% USAGE:
%     M_temporal = MacaqueRetinaM_Temporal_BK
% OUTPUT:
%     M_temporal = impulse response consisting of 300 samples
%                 and with a temporal interval between each
%                 sample of 1ms
%
function M_Temporal = MacaqueRetinaM_Temporal_BK(NSteps)

% number of steps computed ... it MUST be an EVEN number
if nargin==0
    NSteps = 300;
end

% time step (IN S!)
TStep = .001;
MaxFreq = .5*(1/TStep);

% parameters reported in the paper by Benardete and Kaplan
c = .4;
A = 567;
% factor D (in ms. In the original paper it was in s)
D = 2.2/1000;
% if Hs = 1 then the integral of the temporal impulse response is 0
% in the original paper Benardete and Kaplan reported Hs = .98
% in my simulations we put Hs = 1
% Hs = 0.98;
Hs = 1;
C_12 = 0.056;
T0 = 54.60;

% factor tau_S and tau_L (in seconds. In the original paper it was in ms)
tau_S = (T0/(1+(c/C_12)^2))/1000;
tau_L = 1.41/1000;
N_L = 30.30;

% here we compute the impulse response in the Fourier domain
w = 2*pi*linspace(0, MaxFreq, floor(NSteps/2) + 1);
K = exp(-i*w*D) .* (1 - Hs./(1 + i*w*tau_S)) .* ...
	((1./(1+i*w*tau_L)).^N_L);
RemSteps = NSteps - length(K);
FinK = [K, conj(fliplr(K(end-RemSteps:end-1)))];

% ... and here where perform the inverse Fourier Transform
M_Temporal = real(ifft(FinK));

% ... normalize so that the maximum of the response is 1
M_Temporal = M_Temporal/max(abs(M_Temporal));

