
meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.csv');
data = readtable(meta_file);

%% loop over shanks, get RF, average
shanks = unique(data.shank);
numShanks = numel(shanks);

cmap = lines(numShanks);
shankMu = zeros(numShanks, 2);
shankC = zeros(numShanks, 4);

%%
for iShank = 1:numShanks-1

    sessionids = find(data.shank==shanks(iShank));

    ns = numel(sessionids);
    mu = nan(ns,2);
    C = nan(ns,4);
    for isession = 1:ns
        Exp = io.dataFactoryGratingSubspace(sessionids(isession));
        try % get RF if mapping data was present
            [rfsF, rfsC, groupid] = getSpatialRFLocations(Exp, 0);
            mu(isession,:) = rfsF.mu;
            C(isession,:) = rfsF.cov(:)';
        end
    end
    
    if isempty(mu)
        shankMu(iShank,:) = nan;
        shankC(iShank,:) = nan;
        continue
    else        
        shankMu(iShank,:) = nanmedian(mu,1);
        shankC(iShank,:) = nanmedian(C,1);
    end
    
    % add to meta data
    for isession = 1:numel(sessionids)
        thisSess = sessionids(isession);
        
        if isnan(mu(isession,1))
            mu(isession,:) = shankMu(iShank,:);
            C(isession,:) = shankC(iShank,:);
        end
        
        data.retx(thisSess) = mu(isession,1);
        data.rety(thisSess) = mu(isession,2);
        data.retc1(thisSess) = C(isession,1);
        data.retc2(thisSess) = C(isession,2);
        data.retc4(thisSess) = C(isession,4);
%         
%         data.retx(thisSess) = shankMu(iShank,1);
%         data.rety(thisSess) = shankMu(iShank,2);
%         data.retc1(thisSess) = shankC(iShank,1);
%         data.retc2(thisSess) = shankC(iShank,2);
%         data.retc4(thisSess) = shankC(iShank,4);
    end
    
    if ~isnan(shankMu(iShank,1))
        for i = 1:size(mu,1)
            plot.plotellipse(mu(i,:), reshape(C(i,:), [2 2]), 1, 'Color', cmap(iShank,:)); hold on
        end
        plot.plotellipse(median(mu,1), reshape(median(C,1), [2 2]), 1, 'Color', cmap(iShank,:), 'Linewidth', 3);
        drawnow
    end
    
end


%% save it
writetable(data, meta_file);

