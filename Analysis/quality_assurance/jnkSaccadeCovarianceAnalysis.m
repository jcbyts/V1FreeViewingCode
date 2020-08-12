%%

iEx = iEx + 1;
[Exp, S, lfp] = io.dataFactoryGratingSubspace(iEx);

if isempty(lfp.deadChan)
    figure(1); clf
    v = var(lfp.data);

    plot(v); hold on

    thresh = median(v)*.5;
    deadChan = find(v < thresh);
    plot(xlim, thresh*[1 1], 'r')
    plot(deadChan, v(deadChan), 'o')
    lfp.deadChan = deadChan;
end

gstruct = csd.getGamma(lfp);

et = csd.getCSDEventTimes(Exp);
cstruct = csd.getCSD(lfp, et);

figure(iEx); clf
csd.plotCSD_BP(cstruct, gstruct)

%%
[stim, robs, grating] = io.preprocess_grating_subspace_data(Exp);
robs = robs(:,Exp.osp.cids);
%%

X = stim{1};
ev = find(stim{2}) + grating.num_lags_sac_pre;
ev = ev(find(diff(ev) > 20));

NT = size(X,1);

lags = -20:20;
nlags = numel(lags);
inds = ev(:) + lags;
invalid = sum(inds<1 | inds > NT,2)>0;
inds(invalid,:) = [];
NE = size(inds,1);
NC = size(robs,2);
Y = zscore(robs(inds(:),:));
Y = imgaussfilt(Y, [1, 0.001]);


Xevent = reshape(Y, [NE, nlags*NC]);

sta = mean(Xevent);
Xevent = Xevent - sta;

[u,s,v] = svd(cov(Xevent));


sd = diag(s);
figure
plot(sd, '-o')

figure
n = 5;
subplot(1,n+1, 1)
imagesc(1:NC, lags, reshape(sta, [nlags, NC]))
for i = 1:n
    subplot(1,n+1,i+1)
    imagesc(1:NC, lags, reshape(u(:,i), [nlags NC]), [-.1 .1])
end

%%
Y = zscore(robs);
k = reshape(u(:,1), [nlags NC]);
tmp = zeros(size(Y));
for i = 1:size(k,2)
    tmp(:,i) = filter(k(:,i), 1, Y(:,i));
end

figure(1); clf
plot(sum(tmp,2).^2); hold on
plot(stim{2}*15)


figure, 
plot(xcorr(stim{2}, sum(tmp,2).^2, 100))

%%
xlags = 1e3*lags/grating.fs_stim;

figure(4); clf
plot(sd, '-o')

figure(1); clf
plot(xlags, reshape(sta, [nlags, NC]))

figure(2); clf
plot(xlags, reshape(u(:,1), [nlags, NC]))

figure(3); clf
plot(xlags, reshape(u(:,2), [nlags, NC]))



%%
% mean(Exp.vpx.Labels==4)
% numel(io.getValidTrials(Exp, 'Grating'))
