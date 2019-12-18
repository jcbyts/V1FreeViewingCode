
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
sessId = 8;
[Exp, S] = io.dataFactory(sessId);

S.rect = [-10 0 30 30];
%% regenerate data with new parameters?
regenerate = true;
if regenerate
options = {'stimulus', 'Gabor', ...
    'testmode', false, ...
    'eyesmooth', 3, ... % bins
    't_downsample', 2, ...
    's_downsample', 2, ...
    'includeProbe', true, ...
    'correctEyePos', true};

fname = io.dataGenerate(Exp, S, options{:});
end

%% load stimulus

fname = strrep(Exp.FileTag, '.mat', '_Gabor.mat');
dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
fprintf('Loading stimulus\n')
load(fullfile(dataDir, fname))
fprintf('Done\n')



%%
eyePosAtFrame = eyeAtFrame - mean(eyeAtFrame);

NY = size(stim,2)/NX;
nlags = ceil(.05/dt);
NC = size(Robs,2);
%% Test RFs?
dims = [NX NY];

fprintf('Building time-embedded stimulus\n')
tic
spar = NIM.create_stim_params([nlags dims], 'tent_spacing', 2);
X = NIM.create_time_embedding(stim, spar);
X = X-mean(X);
X = zscore(X);
fprintf('Done [%02.2f]\n', toc)

%%
C = X'*X;
%%
ix = valdata == 1 & labels == 1; % & (hypot(eyePosAtFrame(:,1), eyePosAtFrame(:,2)) < 100);% & probeDist > 50;
% ix = valdata == 1 & labels == 1 & eyePosAtFrame(:,1) > 0 & eyePosAtFrame(:,2) > 0;

%%
NTall = size(X,1);
cids = [1 3 4 9 13 18:23 26:28 34:NC];
% cids = 28;
N = numel(cids);
[sacav, ~, saclags] = eventTriggeredAverage(Robs, slist(:,2), [-1 1]*ceil(.2/dt));

figure(10); clf
plot(saclags, sacav/dt)

saclags = [3];
nsaclags = numel(saclags);
stassac = cell(nsaclags,1);
d = size(X,2);
for i = 1:nsaclags
%     inds = intersect(find(valdata), unique(bsxfun(@plus, slist(:,2), saclags(i)+(0:10)))); 
%     inds = unique(slist(:,2) + saclags(i) + (0:2);
    inds = find(ix);
    fprintf('lag: %d, %d valid samples\n', i, numel(inds))
    xy = [(X(inds,:)) ones(numel(inds),1)]'*Robs(inds,cids);
%     stasac{i} = (C + 10e6*eye(d))\xy(1:end-1,:);
    stasac{i} = xy(1:end-1,:);
    figure; clf
    for cc = 1:N
        a = reshape(stasac{i}(:,cc), [nlags prod(dims)]);
        a = (a - min(a(:))) / (max(a(:)) - min(a(:)));
        for ilag = 1:nlags
            subplot(N, nlags, (cc-1)*nlags + ilag, 'align')
            imagesc(xax, yax, reshape(a(ilag,:), [NX NY]), [0 1])
            axis off
            drawnow
        end
    end
%         subplot(6,6,cc, 'align')
%         imagesc(a)
%         axis off
    colormap(parula)
%     end
    drawnow
end


%%
f = @(x) (x);
C = f(X)'*f(X);
fprintf('Running STA...')
xy = [f(X(ix,:)) ones(sum(ix),1)]'*Robs(ix,:);
fprintf('[%02.2f]\n', toc)
xy = xy(1:end-1,:);



%%
d = size(X,2);
stas = (C + 10e5*eye(d))\xy;
% stas = xy;


%% plot
for cc = 1:NC
    a = reshape(stas(:,cc),[nlags prod(dims)]);  
    a = (a - min(a(:))) ./ (max(a(:)) - min(a(:)));
    figure(1); 
    for i = 1:nlags
        subplot(1, nlags, i, 'align')
        imagesc(xax/Exp.S.pixPerDeg/2, yax/Exp.S.pixPerDeg/2, reshape(a(i,:), dims), [0 1])
    end
%     subplot(2,1,1); hold off; plot(a,'b')
    
    title(sprintf('cell %d', cc ))
    drawnow
%     subplot(2,1,2); imagesc(xax/Exp.S.pixPerDeg/2, yax/Exp.S.pixPerDeg/2, reshape(a(3,:), dims))
    input('more?')
end


%%


% params = NIM.create_stim_params([nlags, dims(1), dims(2)], ...
%     'stim_dt', dt, ...
%     'upsampling', 1, ...
%     'tent_spacing', 2);

%% Loop over neurons, fit
NL_types = {'rectlin', 'rectlin'}; % define subunit as linear (note requires cell array of strings)
subunit_signs = [1, -1]; % determines whether input is exc or sup (mult by +1 in the linear case)
x_targs = [1, 1];
% Set initial regularization as second temporal derivative of filter
lambda_d2t = [10e1 10e2];
lambda_d2x = [20e2 10e4];
lambda_l1 = [1 1];
            
f = @(x) x;
NT = length(Robs);

test_inds = ceil(NT*2/5):ceil(NT*3/5);
train_inds = setdiff(1:NT,test_inds);
         
nsubs = 1;
LN0 = NIM( spar, NL_types(1:nsubs), subunit_signs(1:nsubs), ...
    'xtargets', x_targs(1:nsubs), ...
    'd2t', lambda_d2t(1:nsubs), ...
    'l1', lambda_l1(1:nsubs), ...
    'd2x', lambda_d2x(1:nsubs));


%% fit filters
cc = 13;
LN0.subunits(1).filtK = stas(:,cc);
if nsubs > 1
LN0.subunits(2).filtK = stas(:,cc);
end
LN = LN0.fit_filters(Robs(:,cc), {f(X)}, find(ix));


%%
LN.display_model

%%
cc = 13
LN.subunits(1).reg_lambdas.d2x = 2e3;
LN.subunits(1).reg_lambdas.d2t = 10e1;
LN.subunits(1).reg_lambdas.l1 = 20;
LN = LN.fit_filters(Robs(:,cc), {f(X)}, find(ix));
LN.display_model
%%
NIM0 = LN.add_subunits( {'rectlin'}, -1);
NIM0 = NIM0.fit_filters(Robs(:,cc), {f(X)}, find(ix));
NIM0.display_model


%%
sLN = sNIM(LN, 1, spar);
sLN.subunits.reg_lambdas.d2x = 10e4;
sLN.subunits.reg_lambdas.d2t = 10e2;
sLN.subunits.reg_lambdas.l1 = 100;
sLN = sLN.fit_filters(Robs(:,cc), {stim}, find(ix))
sLN.display_model

sLN.add_subunits({'quad'}, 1)
sLN = sLN.fit_filters(Robs(:,cc), {stim})
sLN.display_model


%% GLM: add spike-history term and refit
GLM0 = LN.init_spkhist( 20, 'doubling_time', 5 ); 
GLM0 = GLM0.fit_filters( Robs(:,cc), X, find(ix), 'silent', 0 ); % silent suppresses optimization output

% Display model components
GLM0.display_model('Xstims',X,'Robs',Robs(:,cc))

%%
% Compare likelihoods
[LN.eval_model( Robs(:,cc), X, test_inds )  GLM0.eval_model( Robs(:,cc), X, test_inds )]
% Also available in fit-structures
[LN.fit_props.LL GLM0.fit_props.LL]
	
%% NIM: linear plus suppressive; like Butts et al (2011)
% Add an inhibitory input (with delayed copy of GLM filter and fit 
delayed_filt = NIM.shift_mat_zpad( LN.subunits(1).filtK, 4 );
NIM0 = GLM0.add_subunits( {'rectlin'}, -1, 'init_filts', {delayed_filt} );
NIM0 = NIM0.fit_filters( Robs(:,cc), X, find(ix) );

% Allow threshold of suppressive term to vary
NIM1 = NIM0.fit_filters( Robs(:,cc), X, 'fit_offsets', 1 ); % doesnt make huge difference
% Search for optimal regularization
NIM1 = NIM1.reg_path( Robs(:,c), X, find(ix), test_inds, 'lambdaID', 'd2t' );

% Compare subunit filters (note delay of inhibition)
NIM1.display_subunit_filters()

