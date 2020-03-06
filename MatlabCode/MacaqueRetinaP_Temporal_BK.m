%
% This M-file computes the temporal response of a P-type retinal ganglion
% cell as reported by Benardete and Kaplan (1999)
% Visual Neuroscience (16):355-368
%
% USAGE:
%     P_temporal = MacaqueRetinaP_Temporal_BK
% OUTPUT:
%     P_temporal = impulse response consisting of 300 samples
%                 and with a temporal interval between each
%                 sample of 1ms
%
function [P_Temporal, K_omega] = MacaqueRetinaP_Temporal_BK(Nw)

TimeStep = 1;

if nargin==0
    Nw = 301;
end
omega = linspace(0,pi/TimeStep,Nw/2+1);

% Retina P ON center: from Benardete & Kaplan (1997) I (Table 2) 
% Michele's parameters - means of ON Cells
A = 63.68;
D = 3.47;
tau_s = 31.24;
tau_L = 1.32;
N_L = 36.38;
HS = 0.69;

% Retina P ON center: from Benardete & Kaplan (1997) I (Table 2)
% Gaelle's parameters - median for all cells
% A = 54.57;
% D = 3.5;
% tau_s = 32.27;
% tau_L = 1.55;
% N_L = 31.5;
% HS = 0.73;

K_omega = A * exp(-i*omega*D) .*(1 - HS./(1+i*omega*tau_s)).*((1./(1+i*omega*tau_L)).^N_L);
Spectrum = [K_omega fliplr(conj(K_omega(2:floor(Nw/2))))];

P_Temporal = real(ifft(Spectrum));
P_Temporal = P_Temporal/max(P_Temporal(:));


