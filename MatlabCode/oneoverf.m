function noise =  oneoverf(imsize_x,imsize_y,pxSize)
% CREATES A MASK OF 1/F^2 NOISE
%noise =  nnoise(imsize,pxSize)
% history: % % % % % % % % % % %  
% 
% 14 November: 
% checked temporal profile of mask 
% Magnitude seems to decrease as 1/f2
% exepted for a little plateau initially
% probably due to the sampling error.
% 
% % % % % % % % % % % % % % % % 

% pixel size in degrees
pixAngle = pxSize/60;

% create random noise
im=rand(imsize_x,imsize_y);

% FFT of image and matrix inversion
IM1 = fftshift( fft2(im) );

% magnitude is set to one
IM2 = IM1./abs(IM1);

% builds X and Y of distance from centre
[X Y]=meshgrid( ...
    [ round(size(IM2,2)/2):-1:1 ...
    1:round(size(IM2,2)/2) ],...
    [ round(size(IM2,1)/2):-1:1 ...
    1:round(size(IM2,1)/2) ]);

% rescale matrices in degrees
X = X.*pixAngle;
Y = Y.*pixAngle;

% creates radial matrix
R = sqrt(X.^2 + Y.^2);

% apply 1/F.^2
F = 1./R;
IM3 = IM2 .* F;

% should not this be real already?
n = real( ifft2(ifftshift(IM3)) );
nm = n ./ sqrt(sum( n(:).^2 )) ;
noise = nm;
