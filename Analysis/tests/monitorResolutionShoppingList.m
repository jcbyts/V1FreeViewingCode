
%% Current setup with Propixx at 240Hz

ppd = (Exp.S.screenRect(3)/2)/(atand(Exp.S.screenWidth/Exp.S.screenDistance/2));


%% Acer Predator XB273K Gpbmiipprzx 27"
xpix = 3840;
ypix = 2160;
screenDist = Exp.S.screenDistance;
screenSize = 27;

screenRat = ypix / xpix;
screenWidth = sqrt(screenSize^2 / (1 + screenRat^2));

ppd = (xpix/2) / atand(screenWidth/screenDist/2);

disp(ppd)

%% 1440p options (e.g., from BenQ)
xpix = 2560;
ypix = 1440;
screenDist = Exp.S.screenDistance;
screenSize = 31.5;

screenRat = ypix / xpix;
screenWidth = sqrt(screenSize^2 / (1 + screenRat^2));

ppd = (xpix/2) / atand(screenWidth/screenDist/2);

disp(ppd)