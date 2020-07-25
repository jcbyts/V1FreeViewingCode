
%% 
for iEx = 43:57
    disp(iEx)
    fclose('all');
    [Exp,S] = io.dataFactoryGratingSubspace(iEx);

    sp = io.clip_waveforms(Exp, S);
    disp('Done')
    
end

%%

% cc = cc + 1
figure(1); clf
cmap = lines(NC);
for cc = 1:NC
% w = (squeeze(mean(WF,1)));
w = squeeze(WFmed(cc,:,:));
plot(win/30 + 2*cc, bsxfun(@plus, w, (1:32)*100), 'Color', cmap(cc,:)); hold on

whi = squeeze(WFciHi(cc,:,:));
plot(win/30 + 2*cc, bsxfun(@plus, whi, (1:32)*100), '--', 'Color', cmap(cc,:)); hold on
end

%%

figure, imagesc(w)

%%

S = io.getUnitLocations(Exp.osp);

%%
figure, imagesc(S.templates(:,:,cc))
