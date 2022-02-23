function D = fix_scipy_weirdness(D)
% D = fix_scipy_weirdness(D)

% fix any wierdness from scipy
fields = fieldnames(D);
for f = 1:numel(fields)
%         fprintf('[%s]\n', fields{f})
    if strcmp(fields{f}, 'unit_area')
        sz = size(D.unit_area);
        unit_area = cell(sz(1), 1);
        for i = 1:sz(1)
            unit_area{i} = strrep(D.unit_area(i,:), ' ', '');
        end
        D.unit_area = unit_area;
        continue

    end

    if iscell(D.(fields{f}))
        isnull = strcmp(D.(fields{f}), 'null');
        tmp = cellfun(@double, D.(fields{f}), 'uni', 0);
        tmp = cellfun(@(x) x(1), tmp);
        tmp(isnull) = nan;
        D.(fields{f}) = tmp;
    end
end