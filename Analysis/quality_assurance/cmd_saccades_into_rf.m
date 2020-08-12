addpath ~/Dropbox/MatlabCode/Repos/pdstools/

figdir = fullfile(fileparts(which('addFreeViewingPaths')), 'Figures', 'saccadeSorting');

sessionid = 0;
%% Load data
sessionid = sessionid + 1;
Exp = io.dataFactoryGratingSubspace(sessionid);
exname = strrep(Exp.FileTag, '.mat', '');

%%


figure(1); clf

x = Exp.vpx.smo(:,2);
y =  Exp.vpx.smo(:,3);
valid = hypot(x,y)<5 & [0; diff(hypot(x,y))]~=0;

[C,bx,by] = histcounts2(x(valid),y(valid),50);
bcx = (bx(1:end-1) + bx(2:end))/2;
bcy = (by(1:end-1) + by(2:end))/2;
imagesc(bcx,bcy,log(C')); hold on

plot(0,0, 'r+')
x0 = mode(x(valid));
y0 = mode(y(valid));
plot(x0,y0, 'or')

xlim([-5 5])
ylim([-5 5])

Exp.vpx.smo(:,2)=Exp.vpx.smo(:,2)-x0;
Exp.vpx.smo(:,3)=Exp.vpx.smo(:,3)-y0;

%% get spatial RF
try
[rfsF, rfsC, groupid] = getSpatialRFLocations(Exp, 0);
end
%%
figure(1); clf
set(gcf, 'Color', 'w')
ax = subplot(1,2,1);
imagesc(rfsC.xax, rfsC.yax, rfsC.srf);
colormap gray
hold on
plot(rfsF.xax, rfsF.yax(1)*ones(size(rfsF.xax)), 'r--', 'Linewidth', 2)
plot(rfsF.xax, rfsF.yax(end)*ones(size(rfsF.xax)), 'r--', 'Linewidth', 2)
plot(rfsF.xax(end)*ones(size(rfsF.xax)), rfsF.yax, 'r--', 'Linewidth', 2)
plot(rfsF.xax(1)*ones(size(rfsF.xax)), rfsF.yax, 'r--', 'Linewidth', 2)
xlabel('d.v.a')
ylabel('d.v.a')
axis xy
title(strrep(exname, '_', ' '))

ax2 = subplot(1,2,2);
imagesc(rfsF.xax, rfsF.yax, rfsF.srf);
colormap gray
hold on
plot(rfsF.xax, rfsF.yax(1)*ones(size(rfsF.xax)), 'r--', 'Linewidth', 2)
plot(rfsF.xax, rfsF.yax(end)*ones(size(rfsF.xax)), 'r--', 'Linewidth', 2)
plot(rfsF.xax(end)*ones(size(rfsF.xax)), rfsF.yax, 'r--', 'Linewidth', 2)
plot(rfsF.xax(1)*ones(size(rfsF.xax)), rfsF.yax, 'r--', 'Linewidth', 2)
plot.plotellipse(rfsF.mu, rfsF.cov, 1, 'Color', 'g', 'Linewidth', 2);
xlabel('d.v.a')
ylabel('d.v.a')
axis xy

plot.fixfigure(gcf, 10, [5 2.5])
xlim(ax, rfsC.xax([1 end]));
ylim(ax, rfsC.yax([1 end]));

xlim(ax2, rfsF.xax([1 end]));
ylim(ax2, rfsF.yax([1 end]));


saveas(gcf, fullfile(figdir, [exname '_RFs.pdf']))


%%
% sessionid = 0;
% sessionid = 1;
% 
meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');
data = readtable(meta_file);

%% saccade triggered raster
% sessionid = sessionid + 1;
% Exp = io.dataFactoryGratingSubspace(sessionid);
exname = strrep(Exp.FileTag, '.mat', '');
%
% BackImage
% Grating
validTrials = io.getValidTrials(Exp, 'Grating');
starts = cellfun(@(x) x.START_EPHYS, Exp.D(validTrials));
ends = cellfun(@(x) x.END_EPHYS, Exp.D(validTrials));

ev = Exp.vpx2ephys(Exp.slist(:,1));
vsacix = getTimeIdx(ev, starts, ends);
dx = Exp.vpx.smo(Exp.slist(vsacix,5),2) - Exp.vpx.smo(Exp.slist(vsacix,4),2);
dy = Exp.vpx.smo(Exp.slist(vsacix,5),3) - Exp.vpx.smo(Exp.slist(vsacix,4),3);
ev = ev(vsacix);
NC = numel(Exp.osp.cids);

sorts = {'size'};
for isort = 1:numel(sorts)
    sortby = sorts{isort};
    
    switch sortby
        case 'size'
            [~, rho] = cart2pol(dx, dy);
            d = exp(-rho);
            figOffset = 10;
        case 'angle'
            [th, ~] = cart2pol(dx, dy);
            %         d = cos(th);
            d = exp(-th);
            figOffset = 20;
        case 'rf'
%             mu = [1.2324 -1.5628];
%             mu = rfsF.mu;
            mu = [data.retx(sessionid), data.rety(sessionid)];
            C = [data.retc1(sessionid), data.retc2(sessionid); data.retc2(sessionid) data.retc4(sessionid)];
%             C = rfsF.cov/2;
%             C = [0.8871, 0.2229; 0.2229, 0.5179];
            d = mvnpdf([dx dy], mu, C);
%             d = mvnpdf([dx dy], rfsF(1).mu, rfsF(1).cov);
            d = d / max(d);
            figOffset = 30;
    end
    
    figure(figOffset+1); clf
    
    [~, ind] = sort(d);
    
    bs = 10e-3;
    
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    
    n = 10;
    if strcmp(sortby, 'rf')
        bedges = [0 0.01 0.1 0.5 1];
        n = numel(bedges) - 1;
    else
        bedges = prctile(d(ind), linspace(0, 100, n+1));
    end
    cmap = pdsa.cbrewer('jake', 'rdbu', n);
    % cmap = hsv(n);
    
    for i = 1:n
        ii = d(ind)>bedges(i) & d(ind)<bedges(i+1);
        
        dxs = dx(ind);
        dys = dy(ind);
        plot(dxs(ii), dys(ii), '.','Color', cmap(i,:)); hold on
    end
    title(sortby)
    xlabel('d.v.a')
    ylabel('d.v.a')
    
    pdsa.fixfigure(gcf, 12, [4 4], 'OffsetAxes', false)
    xlim(gca, [-10 10])
    ylim(gca, [-10 10])
    saveas(gcf, fullfile(figdir, [exname '_' sortby 'endpoints.pdf']))
    
    
    figure(figOffset + 2); clf
    for cc = 1:NC
        
        [~,~,bc, ~,wfs] = pdsa.eventPsth(Exp.osp.st(Exp.osp.clu==Exp.osp.cids(cc)), ev(ind), [-.2 .25], bs);
        
        subplot(sx, sy, cc, 'align');
        for i = 1:n
            ii = d(ind)>bedges(i) & d(ind)<bedges(i+1);
            plot(bc, imgaussfilt(mean(wfs(ii,:))/bs, 2), 'Color', cmap(i,:), 'Linewidth', 2); hold on
        end
        plot([0 0], ylim, 'k--')
        title(sprintf('cell: %d', cc), 'fontweight', 'normal')
    end
    
    
    pdsa.suplabel('Time from saccade onset (s)', 'x');
    pdsa.suplabel('Firing Rate (sp/s)', 'y');
    pdsa.suplabel(sortby, 't');
    
    pdsa.fixfigure(gcf, 12, [10 10], 'OffsetAxes', false)
    saveas(gcf, fullfile(figdir, [exname '_' sortby '.pdf']))
end

% saccade end points firing rate
figure(44); clf

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    sptimes = Exp.osp.st(Exp.osp.clu==Exp.osp.cids(cc));
    [wfs, bc] = pdsa.binSpTimes(sptimes, ev, [-.05 .01], bs);
    mw = mean(wfs,2);
    [~, iin] = sort(mw);
    h = scatter(dx(iin), dy(iin),  1 + 10*mw(iin), mw(iin)); hold on
    set(gca, 'Color', .2*[1 1 1])
    plot.plotellipse(mu, C, 1, 'r', 'Linewidth', 2)
    % mw = mean(wfs,2);
    % un = unique(mw);
    % for i = 1:numel(un)
    %     ix = mw==un(i);
    %     plot3(dx(ix),dy(ix),mw(ix), '.'); hold on
    % end
    h.MarkerFaceColor = 'flat';
    xlim([-5 5])
    ylim([-5 5])
    title(cc)
    drawnow
end

set(gcf, 'Color', .4*[1 1 1])
plot.fixfigure(gcf, 10, 2*[sx sy], 'OffsetAxes', false)
saveas(gcf, fullfile(figdir, [exname '_' 'presacEP' '.pdf']))


%%
% %% try to find the RF
% xx = -5:1:5;
% yy = -5:1:5;
% 
% [cnt, edx, edy, indx, indy] = histcounts2(dx,dy,xx,yy);
% [xg,yg] = meshgrid(edx(1:end-1),edy(1:end-1));
% 
% ind = zeros(size(dx));
% good = indx~=0 & indy~=0;
% ind(good) = sub2ind(size(cnt), indy(good), indx(good));
% 
% cc = 0;
% %%
% cc=cc+1;
% if cc > NC
%     cc = 1;
% end
% % [~,~,bc, ~,wfs] = pdsa.eventPsth(Exp.osp.st(Exp.osp.clu==Exp.osp.cids(cc)), ev, [-.2 .25], bs);
% % sptimes = Exp.osp.st(Exp.osp.clu==Exp.osp.cids(cc));
% sptimes = Exp.osp.st;
% [wfs, bc] = pdsa.binSpTimes(sptimes, ev, [-.2 .25], bs);
% 
% Rpre = zeros(size(xg));
% figure(44); clf
% n = numel(unique(ind));
% m0 = mean(wfs); % all
% m0 = m0 - mean(m0);
% m0 = imgaussfilt(m0, 1);
% m0 = .2*m0/max(m0);
% 
% for i = 1:numel(xx)-1
%     for j = 1:numel(yy)-1
%         ix = indx==i & indy==j;
%         m = mean(wfs(ix,:),1);
%         m = m - mean(m);
%         m = imgaussfilt(m, 1);
%         m = .2*m/max(m);
%         
%         x0 = (edx(i)*1);
%         y0 = (edy(j)*1);
%         plot(2*bc+x0, m0+y0, 'Color', .5*[1 1 1]); hold on
%         plot(2*bc+x0, m+y0, 'r'); hold on
%         
%         plot(2*bc+x0, zeros(size(bc))+y0, 'k-');
%         plot([0 0]+x0, [min(m) max(m)]+y0, 'k-');
%         
%         
%         Rpre(j,i) = sum(m(bc<0));
%     end
% end
% 
% title(cc)
% %%
% figure(45); clf
% h = scatter(xg(:),yg(:), 200*ones(numel(xg),1), Rpre(:), 'o');
% h.MarkerFaceColor = 'flat';
% 
% 
% %%
% figure(44); clf
% 
% sx = ceil(sqrt(NC));
% sy = round(sqrt(NC));
% 
% for cc = 1:NC
%     subplot(sx, sy, cc, 'align')
%     sptimes = Exp.osp.st(Exp.osp.clu==Exp.osp.cids(cc));
%     [wfs, bc] = pdsa.binSpTimes(sptimes, ev, [-.1 .01], bs);
%     mw = mean(wfs,2);
%     [~, iin] = sort(mw);
%     h = scatter(dx(iin), dy(iin),  1 + 10*mw(iin), mw(iin)); hold on
%     set(gca, 'Color', .2*[1 1 1])
%     plot.plotellipse(mu, C, 1, 'r', 'Linewidth', 2)
% % mw = mean(wfs,2);
% % un = unique(mw);
% % for i = 1:numel(un)
% %     ix = mw==un(i);
% %     plot3(dx(ix),dy(ix),mw(ix), '.'); hold on
% % end
%     h.MarkerFaceColor = 'flat';
%     xlim([-5 5])
%     ylim([-5 5])
%     title(cc)
%     drawnow
% end
%%
% cc=0;
% %%
% NC = numel(Exp.osp.cids);
% figure(1); clf
% cc = cc+ 1; %cc + 1;
% if cc > NC
%     cc = 1;
% end
% % cc = 13;
% ev = Exp.vpx2ephys(Exp.slist(:,1));
% vsacix = getTimeIdx(ev, starts, ends);
% sum(vsacix)
% 
% cids = Exp.osp.cids;
% 
% dx = Exp.vpx.smo(Exp.slist(vsacix,5),2) - Exp.vpx.smo(Exp.slist(vsacix,4),2);
% dy = Exp.vpx.smo(Exp.slist(vsacix,5),3) - Exp.vpx.smo(Exp.slist(vsacix,4),3);
% ev = ev(vsacix);
% 
% d = mvnpdf([dx dy], rfsF(1).mu, rfsF(1).cov);
% d = d / max(d);
% % sort
% [th, rho] = cart2pol(dx, dy);
% % d = exp(-rho);
% % d = cos(th);
% 
% [ds, ind] = sort(d);
% 
% % psth aligned to saccade onset
% [~,~,bc, ~,wfs] = pdsa.eventPsth(Exp.osp.st(Exp.osp.clu==cids(cc)), ev(ind), [-.2 .25], 1e-3);
% 
% 
% [i,j] = find(wfs);
% 
% tic
% plot.raster(bc(j),(i),4);
% toc
% title(cc)
% axis tight
% 
% figure(2); clf
% 
% %%
% 
% 
% %% all units combined
% figure(2); clf
% bs = 10e-3;
% [~,~,bc, ~,wfs] = pdsa.eventPsth(Exp.osp.st, ev(ind), [-.2 .25], bs);
% imagesc(bc, ds, imgaussfilt(wfs, [20 1]))
% axis xy
% 
% figure(3); clf
% plot(bc, mean(wfs(d(ind)>.05,:))/bs); hold on
% plot(bc, mean(wfs(d(ind)<.01,:))/bs)
% legend(["IN", "OUT"])
% 
% 
% 
% 







