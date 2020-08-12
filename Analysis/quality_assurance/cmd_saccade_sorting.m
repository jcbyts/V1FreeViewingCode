
addpath ~/Dropbox/MatlabCode/Repos/pdstools/
figdir = fullfile(fileparts(which('addFreeViewingPaths')), 'Figures', 'saccadeSorting');

meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');
data = readtable(meta_file);
%% Load session

for sessionid = 46
    Exp = io.dataFactoryGratingSubspace(sessionid);
    exname = strrep(Exp.FileTag, '.mat', '');
    
    STIMULUS = 'Grating';
    sorts = {'size'};
    bs = 10e-3;
    
    % BackImage
    % Grating
    validTrials = io.getValidTrials(Exp, STIMULUS);
    if numel(validTrials) < 5
        continue
    end
    starts = cellfun(@(x) x.START_EPHYS, Exp.D(validTrials));
    ends = cellfun(@(x) x.END_EPHYS, Exp.D(validTrials));
    
    % saccade onset
    ev = Exp.vpx2ephys(Exp.slist(:,1));
    
    % get saccades during the selected stimulus paradigm
    vsacix = getTimeIdx(ev, starts, ends);
    
    % saccade vectors
    dx = Exp.vpx.smo(Exp.slist(vsacix,5),2) - Exp.vpx.smo(Exp.slist(vsacix,4),2);
    dy = Exp.vpx.smo(Exp.slist(vsacix,5),3) - Exp.vpx.smo(Exp.slist(vsacix,4),3);
    ev = ev(vsacix);
    
    NC = numel(Exp.osp.cids);
    
    binnedSpikes = cell(NC,1);
    for cc = 1:NC
        sptimes = Exp.osp.st(Exp.osp.clu==Exp.osp.cids(cc));
        [binnedSpikes{cc},bcenters] = pdsa.binSpTimes(sptimes, ev, [-.2 .25], bs);
    end
    
    % loop over sorts
    for isort = 1:numel(sorts)
        sortby = sorts{isort};
        
        switch sortby
            case 'size'
                [~, rho] = cart2pol(dx, dy);
                d = exp(-rho);
                figOffset = 10;
            case 'angle'
                [th, ~] = cart2pol(dx, dy);
                d = exp(-th);
                figOffset = 20;
            case 'rf'
                mu = [data.retx(sessionid), data.rety(sessionid)];
                C = [data.retc1(sessionid), data.retc2(sessionid); data.retc2(sessionid) data.retc4(sessionid)];
                
                d = mvnpdf([dx dy], mu, C);
                d = d / max(d);
                figOffset = 30;
        end
        
        figure(figOffset+1); clf
        
        [~, ind] = sort(d);
        
        sx = ceil(sqrt(NC));
        sy = round(sqrt(NC));
        
        n = 5;
        if strcmp(sortby, 'rf')
            bedges = [0 0.01 0.1 0.5 1];
            n = numel(bedges) - 1;
        else
            bedges = prctile(d(ind), linspace(0, 100, n+1));
        end
        cmap = pdsa.cbrewer('jake', 'rdbu', n);
        
        ns = zeros(n,1);
        for i = 1:n
            ii = d>bedges(i) & d<bedges(i+1);
            ns(i) = sum(ii);
            plot(dx(ii), dy(ii), '.','Color', cmap(i,:)); hold on
        end
        
        title(sortby)
        xlabel('d.v.a')
        ylabel('d.v.a')
        
        % save figure
        pdsa.fixfigure(gcf, 12, [4 4], 'OffsetAxes', false)
        xlim(gca, [-10 10])
        ylim(gca, [-10 10])
        saveas(gcf, fullfile(figdir, ['endpoints' sortby '_' exname '.pdf']))
        
        
        figure(figOffset + 2); clf
        for cc = 1:NC
            wfs = binnedSpikes{cc};
            
            subplot(sx, sy, cc, 'align');
            for i = 1:n
                ii = d>bedges(i) & d<bedges(i+1);
                firingRate = mean(wfs(ii,:))/bs;
                plot(bcenters, imgaussfilt(firingRate, 2), 'Color', cmap(i,:), 'Linewidth', 2); hold on
            end
            plot([0 0], ylim, 'k--')
            title(sprintf('cell: %d', Exp.osp.cids(cc)), 'fontweight', 'normal')
        end
        
        pdsa.suplabel('Time from saccade onset (s)', 'x');
        pdsa.suplabel('Firing Rate (sp/s)', 'y');
        pdsa.suplabel(sortby, 't');
        
        % save figure
        pdsa.fixfigure(gcf, 12, [10 10], 'OffsetAxes', false)
        saveas(gcf, fullfile(figdir, ['firingRate_' sortby '_' exname '.pdf']))
    end
    
%     % saccade end points firing rate
%     figure(44); clf
%     
%     sx = ceil(sqrt(NC));
%     sy = round(sqrt(NC));
%     
%     for cc = 1:NC
%         subplot(sx, sy, cc, 'align')
%         wfs = binnedSpikes{cc}(:,bcenters > -.05 & bcenters < 0.02);
%         mw = mean(wfs,2); % mean pre-saccadic firing rate
%         
%         [~, iin] = sort(mw);
%         h = scatter(dx(iin), dy(iin),  1 + 10*mw(iin), mw(iin)); hold on
%         h.MarkerFaceColor = 'flat';
%         
%         % set background
%         set(gca, 'Color', .2*[1 1 1])
%         
%         % plot RF
%         plot.plotellipse(mu, C, 1, 'r', 'Linewidth', 2)
%         
%         
%         xlim([-2 2]*hypot(mu(1), mu(2)))
%         ylim([-2 2]*hypot(mu(1), mu(2)))
%         title(cc)
%         drawnow
%     end
%     
%     set(gcf, 'Color', .4*[1 1 1])
%     plot.fixfigure(gcf, 10, 2*[sx sy], 'OffsetAxes', false)
%     saveas(gcf, fullfile(figdir, ['presacEP_' exname '.pdf']))
    
end
%%

cc = cc+ 1; 
if cc > NC
    cc = 1;
end
figure(1); clf
plot( (mean(binnedSpikes{cc})-mean(binnedSpikes{cc}(:)))/std(binnedSpikes{cc}(:)))
title(cc)
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







