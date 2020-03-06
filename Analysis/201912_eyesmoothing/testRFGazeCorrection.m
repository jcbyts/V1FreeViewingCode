
%% add paths

user = 'jakework';
addFreeViewingPaths(user);

switch user
    case 'jakework'
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\NIMclass
        addpath C:\Users\Jake\Dropbox\MatlabCode\Repos\sNIMclass
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\L1General'))    
        addpath(genpath('C:\Users\Jake\Dropbox\MatlabCode\Repos\minFunc_2012'))  
    case 'jakelaptop'
        addpath ~/Dropbox/MatlabCode/Repos/NIMclass/
        addpath ~/Dropbox/MatlabCode/Repos/sNIMclass/
        addpath(genpath('~/Dropbox/MatlabCode/Repos/L1General/'))    
        addpath(genpath('~/Dropbox/MatlabCode/Repos/minFunc_2012/'))  
end

%% load data
[~, S] = io.dataFactory(8);

% overwrite with original
Exp = load('Data\L20191205_bkp.mat');

% correct based on calibraiton trials
eyePos = io.getCorrectedEyePos(Exp, 'usebilinear', true);

% run RF based correction once
eyePos2 = io.getCorrectedEyeposRF(Exp, S, eyePos);

drawnow
%%
% run it again
[eyePos3, ep4] = io.getCorrectedEyeposRF(Exp, S, eyePos2);

%% load data
% [Exp, S] = io.dataFactory(8);
% 

eyePosA = eyePos3;

% run RF based correction once
eyePosB = io.getCorrectedEyeposRF(Exp, S, eyePosA);