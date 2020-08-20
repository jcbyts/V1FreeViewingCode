
%% Load dataset 

sessid = 56;
[Exp, S] = io.dataFactoryGratingSubspace(sessid, 'spike_sorting', 'jrclustwf');

%%
field = 'raw0';
tt = Exp.vpx.(field)(:,1);
xpos = Exp.vpx.(field)(:,2);
ypos = Exp.vpx.(field)(:,3);

figure(1); clf
plot(tt, hypot(xpos, ypos), '-ok', 'MarkerSize', 2)

%% How many single neurons do we have?
W = io.get_waveform_stats(Exp.osp);
tmp = io.getUnitLocations(Exp.osp);
%%
figure(1); clf
NC = numel(W);
cmap = lines(NC);

SU = false(NC,1);

for cc = 1:NC
    nw = norm(W(cc).waveform(:,3));
    SU(cc) = nw > 20 & W(cc).isiL > .1 & W(cc).isiL < 1 & W(cc).uQ > 5 & W(cc).isi(200) < 0;
	
    nts = size(W(1).waveform,1);
    xax = linspace(0, 1, nts) + cc;
    if SU(cc)
        clr = cmap(cc,:);
    else
        clr = .5*[1 1 1];
    end
    plot(xax, W(cc).waveform + W(cc).spacing + W(cc).depth, 'Color', clr, 'Linewidth', 2); hold on
    
end
xlabel('Unit #')
ylabel('Depth')
cids = arrayfun(@(x) x.cid, W(SU));
plot.fixfigure(gcf, 12, [8 4])

%%
figure(1); clf
histogram(arrayfun(@(x) norm(x.waveform(:,3)), W), 100)


%%
figure(3); clf
plotWaveforms(W, cids)
xlabel('Unit #')
ylabel('Depth')
%%
figure(2); clf
NC = numel(cids);
cmap = lines(NC);
ax = plot.tight_subplot(NC, NC, 0.001, 0.001, 0.001);
for ii = 1:NC
    sptimes1 = Exp.osp.st(Exp.osp.clu==cids(ii));
    for jj = 1:NC
        sptimes2 = Exp.osp.st(Exp.osp.clu==cids(jj));
        set(gcf, 'currentaxes', ax((ii-1)*NC + jj))
        if jj >= ii
            
            [xc, lags] = crossCorrelation(sptimes1, sptimes2, 'numLags', 100, 'binSize', 1e-3);
            if ii==jj
                xc(lags==0) = 0;
            end
%             subplot(NC, NC, (ii-1)*NC + jj, 'align')
%             set(gcf, 'currentaxes', ax((ii-1)*NC + jj))
            bar(lags, imgaussfilt(xc, 2), 'FaceColor', cmap(jj,:), 'EdgeColor', 'none')
            drawnow
        end
        axis off
    end
end

%%


function plotWaveforms(W, cids)
    NC = numel(cids);
    cid0 = arrayfun(@(x) x.cid, W);
    cmap = lines(NC);
    
    for cc = 1:NC
        
        cid = cids(cc)==cid0;
        nts = size(W(cid).waveform,1);
        xax = linspace(0, 1, nts) + cc;

        plot(xax, W(cid).waveform + W(cid).spacing + W(cid).depth, 'Color', cmap(cc,:), 'Linewidth', 2); hold on
    
    end

end