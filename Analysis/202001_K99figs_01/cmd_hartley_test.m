% Use the Grating subspace to measure the subspace receptive fields during
% free-viewing full-field gratins


%% Load data
[Exp, S] = io.dataFactory(12);

validTrials = io.getValidTrials(Exp, 'Grating');

%%
[stim, Robs, opts, params, kx, ky] = io.preprocess_grating_subspace_data(Exp);
NT = size(stim{1},1);
NC = size(Robs,2);

test_inds = ceil(NT*2/5):ceil(NT*3/5);
train_inds = setdiff(1:NT,test_inds);
test_is_frozen = false;

Xstim = NIM.create_time_embedding(stim{1}, params(1));

Xd = [Xstim ones(NT,1)];
C = Xd'*Xd;
xy = Xd'*Robs;

wls = (C + 1e2*eye(size(C,2)))\xy;
sta = wls(1:end-1,:);
%%
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(1); clf

cids = 1:NC;
for ci = 1:NC
    subplot(sx, sy, ci, 'align')
    cc = cids(ci);

    a = reshape(sta(:,cc), [opts.num_lags_stim, prod(opts.dim)]);
    [u,s,v] = svd(a);
    I = reshape(v(:,1), opts.dim);
%     I = reshape((a(24,:)), opts.dim);
    plot(opts.oris, I)
%     imagesc(a)
%     ylim([0 40])
    title(cc)
end




%% try some NIM
cc = mod(cc+1, NC); cc(cc==0)=1;
MODEL_DIR = 'jnk';
% tag = fullfile(MODEL_DIR, get_tag(S.processedFileName, opts, cc));
tag = 'jnk';
LN0 = prf.fit_LN(Robs(:,cc), stim, params, train_inds, test_inds, tag, 'reg_path', 0, 'overwrite', 1, 'verbose', true, 'nosave', 1);
LN0.display_model

%%
NIM0 = prf.fit_add_quad_subunit(LN0, Robs(:,cc), stim, train_inds, test_inds, tag, 'overwrite', 0);
NIM0.display_model
        
if isnan(NIM0.fit_props.LL)
    NIM0 = LN0;
end

[O,OG,Xstim] = prf.fit_add_offset_gain(NIM0, Robs(:,cc), stim, opts, train_inds, test_inds, tag, 'reg_path', 1, 'overwrite', false, 'verbose', true);
        
OG.display_model        

nlags = 30;
Xd = makeStimRows(stim{1}, nlags);
NT = size(stim{1},1);
sta = [Xd ones(NT,1)]'*Robs;
sta = sta(1:end-1,:);

figure
plot(sta)
% sta = sta./sum(X)';
%%


%%
% 


