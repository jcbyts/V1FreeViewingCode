function run_analyses(fdir, fout, flist, ifile)


fprintf('Running analysis for [%s]\n', flist(ifile).name)
if exist(fullfile(fout, flist(ifile).name), 'file')
    return
end
% load Dstruct file
D = load(fullfile(fdir, flist(ifile).name));

% fix any wierdness from scipy
fields = fieldnames(D);
for f = 1:numel(fields)
    fprintf('[%s]\n', fields{f})
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
        D.(fields{f}) = cell2mat(D.(fields{f}));
    end
end

if isfield(D, 'unit_area')
    if ~any(strcmp(D.unit_area, 'VISp'))
        return
    end
end


%     try
[Stim, opts, Rpred_ind, Rpred] = do_regression_analysis_inspect_individual_cells_dfs(D);

if ~isempty(Stim)
disp('Saving...')
save(fullfile(fout, flist(ifile).name), '-v7.3', 'Stim', 'opts', 'Rpred', 'Rpred_ind')
disp('Done')
end