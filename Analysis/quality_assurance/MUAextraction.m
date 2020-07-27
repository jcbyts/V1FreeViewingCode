

for iEx = 37:55

[Exp, S, lfp, mua] = io.dataFactoryGratingSubspace(iEx);

end


%% test single session extraction from server
ops = io.loadOps(S.rawFilePath);

if numel(ops) > 1
    ops = ops(1);
end

% get MUA

[MUA, timestamps] = io.getMUA(ops);