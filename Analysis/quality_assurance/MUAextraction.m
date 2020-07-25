

iEx = 5;

[Exp, S] = io.dataFactoryGratingSubspace(iEx);

ops = io.loadOps(S.rawFilePath);

if numel(ops) > 1
    ops = ops(1);
end

%% get MUA

[MUA, timestamps] = io.getMUA(ops);