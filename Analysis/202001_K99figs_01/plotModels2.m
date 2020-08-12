model = load('Data/export1231_csn1.mat');


% out = load('Data/gmodsac0_output.mat');

%%

figure(1); clf
imagesc(model.ws10); colormap jet

NX = 24;
nlags = 10;

isub = isub+1;
w = reshape(model.ws00(:,isub), [nlags, NX, NX]);
w = (w - min(w(:))) / (max(w(:)) - min(w(:)));
for ilag = 1:nlags
    subplot(1,nlags,ilag)
    imagesc(squeeze(w(ilag,:,:)), [0 1])
end
colormap gray
% model.ws00*model.ws01


%%
figure(1); clf
set(gcf, 'Color', 'w')

% pos = ax.Position;

% [~, wind] = sort(sum(-model.ws02,2), 'descend');

nsub = size(model.ws01,2);
wind = 1:nsub;
for isub = 1:nsub
    subplot(4, nsub, isub, 'align')
%     subplot(nsub, nsub, (isub-1)*nsub + 1, 'align')
    w = model.ws01(1:2:end,wind(isub));
%     w2 = model.ws01(2:2:end,wind(isub));
    w = zscore(w);
%     w = zscore(w2);
    imagesc(reshape(w, [40 40]), [-1 1]*9)
    axis off
    
%     subplot(4, nsub, 2*nsub + isub, 'align')
    
%     subplot(nsub, nsub, (isub-1)*nsub + 2, 'align')
    subplot(4, nsub, nsub + isub, 'align')
    xax = -10:10;
    plot(xax([1 end]), [0 0], 'k'); hold on
    plot(xax, max(xax, 0), 'r');
    
    plot([0 0],[-1 10], 'k')
    axis off
end
colormap(parula)
%%
% ax = subplot(nsub, nsub, (0:nsub-1)'*nsub + (3:4), 'align');
% ax.Position(1) = ax.Position(1) - 0.005; 
subplot(4,1,3:4)
NC = size(model.ws20,2);
% cmap = flipud(pdsa.cbrewer('jake', 'rdbu', 20));
cmap = lines(NC);

for cc = 36 %mod(cc + 1, NC); cc(cc==0) = 1;
    wts = -model.ws20(wind,cc);
    wts = wts / norm(wts);
%     wts = (wts - min(wts)) / (max(wts) - min(wts));
    plot(wts, '-o', 'Color', cmap(cc,:)); hold on
    
%    
%     wts = -10:9;
%     xpos = repmat([0 1], nsub, 1);
%     cinds = plot.getColors(wts, nan, nan, cmap);
%     wts = abs(wts);
%     wts = (wts - min(wts)) / (max(wts) - min(wts));
%     for isub = 1:nsub
%         plot([0 1], [isub cc/NC*nsub], 'Color', cinds(isub,:), 'Linewidth', exp(wts(isub)) + .01); hold on
%     end

end
plot(xlim, [0 0], 'k')
xlim([0.5 16.5])
% ylim([0.5 20.5])
% axis off
% axis ij

    
%%

%%
iix =1:numel(out.sac_off);
soff = find(out.sac_off(iix));
son = find(out.sac_on(iix));

NC = size(out.Robs,2);
figure(1); clf
cc = mod(cc + 1, NC); cc(cc==0) = 1;
cc = 20;

R = out.Robs(:,cc);
% predRate = nan(size(out.Robs(:,1)));
predRate = out.predrate(:,cc);

plot(imboxfilt(out.Robs(:,cc), 9));  hold on
plot(imboxfilt(predRate, 3));
% title([cc corr(out.predrate(:,cc), out.Robs(out.valdata,cc))]);


% [sprate, bc] = pdsa.empiricalNonlinearity(out.Robs(:,cc), out.predrate(:,cc), 100);
% figure(3); clf
% plot(bc, sprate, '.'); hold on
% plot(xlim, xlim, 'k')


[an0, ~, xax] = eventTriggeredAverage(R, son, [-40 40]);
[an1] = eventTriggeredAverage(predRate, son, [-40 40]);

figure(2); clf
plot(xax, an0, 'k'); hold on
plot(xax, an1, 'r')



% eventTriggeredAverage(Robs(:,cc)
%%
figure(1); clf
C1 = corr(out.predrate);
C0 = corr(out.Robs);

plot(C0(:), C1(:), '.')

