%
% Compute the spatial kernels for M-type retinal ganglion cells
% between 0 and 10 degrees of visual eccentricity
%
% Data for modeling are taken from Croner and Kaplan
% Vision Research (35)1:7-24 (1995)
% 
% USAGE :
%    [Fc, Fs] =
%        MacaqueRetinaM_Spatial_CK(PixelAngle, Eccentricity, GridX, GridY)
% OUTPUT :
%    Fc = positive part of the spatial "mexican hat"
%    Fs = negative part of the spatial "mexican hat"
% INPUT :
%    PixelAngle = width of one pixel (in DEGREES!!!)
%    Eccentricity = retinal eccentricity of the ganglion cell
%    GridX = (opt.) Width along the x axis of the generated spatial profile
%    GridY = (opt.) Width along the y axis of the generated spatial profile
%

function [Fc, Fs] = MacaqueRetinaM_Spatial_CK(PixelAngle, Eccentricity, GridX, GridY)

% both rc and rs are originally in degrees and we transform them into
% minutes of arc.
if (Eccentricity < 10 ) 
	rc = 0.10*60/(60*PixelAngle);
	rs = 0.72*60/(60*PixelAngle);
	Kc = 148;
	Ks = 1.1;
elseif ( (Eccentricity >= 10) & (Eccentricity < 20 ) ) 
	rc = 0.18*60/(60*PixelAngle);
	rs = 1.19*60/(60*PixelAngle);
	Kc = 115;
	Ks = 2;
elseif ( (Eccentricity >=  20) & (Eccentricity < 30 ) ) 
	rc = 0.23*60/(60*PixelAngle);
	rs = 0.58*60/(60*PixelAngle);
	Kc = 63.8;
	Ks = 1.6;
end

% initialize the output
Fc = zeros(GridX, GridY);
Fs = zeros(GridY, GridY);

% Spatial response profile
CenterX = floor(GridX/2) + 1;
CenterY = floor(GridY/2) + 1;

[XX, YY] = meshgrid(1:GridX, 1:GridY);

% ...here we go with the final step
Fc = Kc*exp( -((XX-CenterX).^2+(YY-CenterY).^2)/(rc^2));
Fs = -Ks*exp( -((XX-CenterX).^2+(YY-CenterY).^2)/(rs^2));


