function res = mergeStruct(x,y)

if isstruct(x) && isstruct(y)
    res = x;
    names = fieldnames(y);
    for fnum = 1:numel(names)
        if isfield(x,names{fnum})
            res.(names{fnum}) = mergeStruct(x.(names{fnum}),y.(names{fnum}));
        else
            res.(names{fnum}) = y.(names{fnum});
        end
    end
else
    res = y;
end