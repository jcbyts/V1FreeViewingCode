%
% Compute the spatial kernels for P-type retinal ganglion cells
% between 0 and 10 degrees of visual eccentricity
%
% Data for modeling are taken from Croner and Kaplan
% Vision Research (35)1:7-24
% 
% USAGE :
%    [Fc, Fs] =0
%        MacaqueRetinaP_Spatial_CK(PixelAngle, Eccentricity, GridX, GridY)
% OUTPUT :
%    Fc = positive part of the spatial "mexican hat"
%    Fs = negative part of the spatial "mexican hat"
% INPUT :
%    PixelAngle = width of one pixel (in DEGREES!!!)
%    Eccentricity = retinal eccentricity of the ganglion cell
%    GridX = (opt.) Width along the x axis of the generated spatial profile
%    GridY = (opt.) Width along the y axis of the generated spatial profile
%

function [Fc, Fs] = MacaqueRetinaP_Spatial_CK(PixelAngle, Ecc, GridX, GridY)

% both rc and rs are originally in degrees and we transform them into
% minutes of arc.
if Ecc < 5
	Kc = 325.2;
	Ks = 4.4;
	rc = 0.03 * 60 / (60*PixelAngle); % arcmin
	rs =  0.18 * 60 / (60*PixelAngle);  % arcmin
elseif Ecc < 10
	Kc = 114.7;
	Ks = 0.7;
	rc = 0.05 * 60 / (60*PixelAngle); % arcmin
	rs =  0.43 * 60 / (60*PixelAngle);  % arcmin
elseif Ecc < 20
	Kc = 77.8;
	Ks = 0.6;
	rc = 0.07 * 60 / (60*PixelAngle); % arcmin
	rs =  0.54 * 60 / (60*PixelAngle);  % arcmin
elseif Ecc < 30
	Kc = 57.2;
	Ks = 0.8;
	rc = 0.09 * 60 / (60*PixelAngle); % arcmin
	rs =  0.73 * 60 / (60*PixelAngle);  % arcmin
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


