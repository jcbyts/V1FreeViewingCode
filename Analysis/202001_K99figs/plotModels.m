model = load('Data/snim1export.mat');
out = load('Data/snim1_output.mat');


%%
figure(1); clf
set(gcf, 'Color', 'w')

% pos = ax.Position;

[~, wind] = sort(sum(-model.ws02,2), 'descend');

nsub = size(model.ws01,2);
for isub = 1:nsub
    subplot(4, nsub, isub, 'align')
%     subplot(nsub, nsub, (isub-1)*nsub + 1, 'align')
    w = model.ws01(1:end,wind(isub));
    w = zscore(w);
    imagesc(reshape(w, [20 20]), [-1 1]*5)
    axis off
    
%     subplot(nsub, nsub, (isub-1)*nsub + 2, 'align')
    subplot(4, nsub, nsub + isub, 'align')
    xax = -10:10;
    plot(xax([1 end]), [0 0], 'k'); hold on
    plot(xax, max(xax, 0), 'r');
    
    plot([0 0],[-1 10], 'k')
    axis off
end
colormap(gray)

% ax = subplot(nsub, nsub, (0:nsub-1)'*nsub + (3:4), 'align');
% ax.Position(1) = ax.Position(1) - 0.005; 
subplot(4,1,3:4)
NC = size(model.ws02,2);
% cmap = flipud(pdsa.cbrewer('jake', 'rdbu', 20));
cmap = lines(NC);
[~, dind] = sort(Exp.osp.clusterDepths, 'ascend');

for cc = [1 8 32] %mod(cc + 1, NC); cc(cc==0) = 1;
    wts = -model.ws02(wind,cc);
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
xlim([0.5 20.5])
% ylim([0.5 20.5])
% axis off
% axis ij

    
%%

%%

soff = find(out.sac_off);
son = find(out.sac_on);
NC = size(out.Robs,2);
figure(1); clf
cc = mod(cc + 1, NC); cc(cc==0) = 1;

% plot(imgaussfilt(out.predrate(:,cc), 3)); hold on
% plot(imgaussfilt(out.Robs(:,cc), 3))
plot(mean(out.predrate,2)); hold on
plot(mean(out.Robs,2));
title([cc corr(out.predrate(:,cc), out.Robs(:,cc))]);


[sprate, bc] = pdsa.empiricalNonlinearity(out.Robs(:,cc), out.predrate(:,cc), 100);
figure(3); clf
plot(bc, sprate, '.'); hold on
plot(xlim, xlim, 'k')


[an0, ~, xax] = eventTriggeredAverage(out.Robs(:,cc), son, [-40 40]);
[an1] = eventTriggeredAverage(out.predrate(:,cc), son, [-40 40]);

figure(2); clf
plot(xax, an0, 'k'); hold on
plot(xax, an1, 'r')



% eventTriggeredAverage(Robs(:,cc)
%%
figure(1); clf
C1 = corr(out.predrate);
C0 = corr(out.Robs);

plot(C0(:), C1(:), '.')

