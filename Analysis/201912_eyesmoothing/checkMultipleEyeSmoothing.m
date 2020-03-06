
%% add paths

user = 'jakework';
addFreeViewingPaths(user);


%% load data

for sessId = 8%9:-1:1
    [Exp, S] = io.dataFactory(sessId);
    
    
    for iSm = [1 3 5]
        options = {'stimulus', 'Gabor', ...
            'testmode', 50, ...
            'eyesmooth', iSm, ... % bins
            't_downsample', 2, ...
            's_downsample', 2, ...
            'includeProbe', false};
        
        
        fname = io.dataGenerate(Exp, S, options{:});
        
        % --- see that it worked
        dataDir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
        load(fullfile(dataDir, fname))
        
        
        figname = strrep(fname, '.mat', '');
        
        % STA on a subset of units
        figure(2); clf
        ix = valdata == 1 & labels == 1 & probeDist > 50;
        nk = ceil(.1/dt);
        X = makeStimRows(stim, nk);
        
        for k = 1:min(36, size(Robs,2))
            
            sta = [X(ix,:) ones(sum(ix),1)]' * Robs(ix,k);
            sta = reshape(sta(1:end-1), nk, []);
            subplot(6, 6, k, 'align')
            imagesc(sta)
            set(gca, 'XTickLabel', '', 'YTickLabel', '')
        end
        
        plot.fixfigure(gcf, 10, [10 10], 'OffsetAxes', false)
        
        hx = plot.suplabel('space (x by y)', 'x'); set(hx, 'FontSize', 14)
        hy = plot.suplabel('time (runs up)', 'y');  set(hy, 'FontSize', 14)
        ht = plot.suplabel('STA in on raw pixels (Gabors)', 't');  set(ht, 'FontSize', 14)
        
        saveas(gcf, fullfile('Figures', 'eyesmooth', sprintf('%s_sta_eyesmooth_%d.pdf', figname, iSm)))
        
        % STA with squared pixel values
        figure(2); clf
        ix = valdata == 1 & labels == 1 & probeDist > 50;
        for k = 1:min(36, size(Robs,2))
            
            sta = [X(ix,:).^2 ones(sum(ix),1)]' * Robs(ix,k);
            sta = reshape(sta(1:end-1), nk, []);
            subplot(6, 6, k, 'align')
            imagesc(sta)
            set(gca, 'XTickLabel', '', 'YTickLabel', '')
            
        end
        
        plot.fixfigure(gcf, 10, [10 10], 'OffsetAxes', false)
        
        hx = plot.suplabel('space (x by y)', 'x'); set(hx, 'FontSize', 14)
        hy = plot.suplabel('time (runs up)', 'y');  set(hy, 'FontSize', 14)
        ht = plot.suplabel('STA in on squared pixels (Gabors)', 't');  set(ht, 'FontSize', 14)
        
        saveas(gcf, fullfile('Figures', 'eyesmooth', sprintf('%s_staQuad_eyesmooth_%d.pdf', figname, iSm)))
        
    end
end
