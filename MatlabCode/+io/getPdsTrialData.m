function data = getPdsTrialData(PDS)
% getPdsTrialData merges the conditions and data and outputs the parameters
% and data for each trial.
% The hierarchy of parameters is like this:
% data -> conditions -> initialParameters

if iscell(PDS)
    nPds = numel(PDS);
    data = cell(nPds,1);
    for k = 1:nPds
        data{k} = io.getPdsTrialData(PDS{k});
    end
    
    fnames = {};
    for k = 1:nPds
        fnames = union(fnames, fieldnames(data{k}));
    end
    
    n = numel(fnames);
    sargs = [fnames cell(n, 1)]';
    nTrials = cellfun(@numel, data);
    tr0 = [0; cumsum(nTrials(1:end-1))];
    tmp = repmat(struct(sargs{:}), sum(nTrials), 1);
    
    for k = 1:nPds
        for j = 1:nTrials(k)
            for f = 1:n
                if isfield(data{k}(j), fnames{f})
                    tmp(tr0(k)+j).(fnames{f}) = data{k}(j).(fnames{f});
                end
            end
            tmp(tr0(k)+j).pdsIndex = k;
            if isfield(PDS{k}, 'PTB2OE')
                tmp(tr0(k)+j).PTB2OE = PDS{k}.PTB2OE;
            end
        end
    end
    
    data = tmp;
    
else
    if ~isempty(PDS.conditions)
        if any(cellfun(@isempty, PDS.conditions))
            for ii = 1:numel(PDS.data)
                PDS.conditions{ii}.Nr = ii;
            end
        end
        
        A = cellfun(@(x) mergeStruct(PDS.initialParametersMerged, x), PDS.conditions, 'uni', 0);
    else
        A = repmat({PDS.initialParametersMerged}, 1, numel(PDS.data));
    end
    fields = fieldnames(PDS.initialParametersMerged);
    nTrials = numel(PDS.data);
    
    for iTrial = 1:nTrials
        fields = union(fields, fieldnames(A{iTrial}));
        fields = union(fields, fieldnames(PDS.data{iTrial}));
    end
    
    sargs = [fields cell(numel(fields), 1)]';
    data = repmat(struct(sargs{:}), nTrials, 1);
    
    for iTrial = 1:nTrials
        data(iTrial) = mergeStruct(data(iTrial), A{iTrial});
        data(iTrial) = mergeStruct(data(iTrial), PDS.data{iTrial});
    end
    
    % overload broken things
    for iTrial = 1:nTrials
        data(iTrial).pldaps.iTrial = iTrial;
    end
    
    data(iTrial).pdsIndex = 1;
    if isfield(PDS, 'PTB2OE')
        data(iTrial).PTB2OE = PDS.PTB2OE;
    end
    
end

% reorder trials so they are sequential
tstarts = arrayfun(@(x) x.timing.flipTimes(1), data);
[~, ind] = sort(tstarts);
data = data(ind);


% loop over fields and correct any empty modules
fnames = fieldnames(data);
for i = 1:numel(fnames)
%     fprintf('%s\n', fnames{i})
    % check for missing "use" fields
    ixuse = arrayfun(@(x) isfield(x.(fnames{i}), 'use'), data);
    ixempty = ~ixuse;
    if any(ixuse)
        emptyTrials = find(ixempty);
        for j = emptyTrials(:)'
            data(j).(fnames{i}).use = false;
        end
    end
    
%     ixempty = arrayfun(@(x) isempty(x.(fnames{i})), data);
%     ixuse   = arrayfun(@(x) isfield(x, fnames{i}), data);
%     if any(ixempty) && any(ixuse) % module is empty on some trials
%         emptyTrials = find(ixempty);
%         for j = emptyTrials(:)'
%             data(j).(fnames{i}).use = false;
%         end
%     end
    
    
end
