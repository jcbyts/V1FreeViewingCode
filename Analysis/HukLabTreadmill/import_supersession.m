function import_supersession(subj, fpath)
% Create a Super Session file for a particular subject

% --- get path
if ~exist('fpath', 'var')
    fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
    fpath = fullfile(fpath, 'gratings');
end

% addpath(fullfile(fpath, 'PLDAPStools')) % add PLDAPStools

if nargin < 1
    subj = 'gru';
end

validSubjs = {'gru', 'brie', 'allen'};
assert(ismember(subj,validSubjs), sprintf("import_supersession: subj name %s is not valid", subj))

%% --- loop over sessions, build big spike struct

Dbig = struct('GratingDirections', [], 'GratingFrequency', [], ...
    'GratingOffsets', [], 'GratingOnsets', [], 'GratingSpeeds', [], ...
    'GratingContrast', [], ...
    'eyeLabels', [], 'eyePos', [], 'eyeTime', [], ...
    'frameTimes', [], 'framePhase', [], 'frameContrast', [], ...
    'treadSpeed', [], 'treadTime', [], ...
    'sessNumSpikes', [], 'sessNumGratings', [], ...
    'sessNumTread', [], 'sessNumEye', [], 'spikeTimes', [], 'spikeIds', []);

Dbig.units = {};
%%

fprintf('Loading and aligning spikes with behavior\n')

unique_sessions = {'gru_20211217', 'allen'}; % sessions for which the unit numbers should not be combined

flist = dir(fullfile(fpath, [subj '*']));
startTime = 0; % all experiments start at zero. this number will increment as we concatenate sessions
newOffset = 0;
timingFields = {'GratingOnsets', 'GratingOffsets', 'spikeTimes', 'treadTime', 'eyeTime', 'frameTimes'};
nonTimingFields = {'GratingDirections', 'GratingFrequency', 'GratingSpeeds', 'eyeLabels', 'eyePos', 'treadSpeed', 'spikeIds', 'sessNumSpikes', 'sessNumGratings', 'sessNumTread', 'sessNumEye', 'framePhase', 'GratingContrast', 'frameContrast'};

fprintf('Looping over %d sessions\n', numel(flist))

for iSess = 1:numel(flist)
    fprintf('%d/%d session [%s]\n', iSess, numel(flist), flist(iSess).name)
    if contains(flist(iSess).name, 'D_all')
        disp('skipping existing supersession')
        continue
    end

    D = load(fullfile(flist(iSess).folder, flist(iSess).name));

    if isempty(D.frameContrast)
        disp('frameContrast is empty. Skipping...')
        continue
    end
    
    D = fix_scipy_weirdness(D);

    if isfield(D, 'unit_area')
        cids = unique(D.spikeIds);
        visp = strcmp(D.unit_area, 'VISp');
        
        iix = ismember(D.spikeIds, cids(visp));
        D.spikeTimes = D.spikeTimes(iix);
        D.spikeIds = D.spikeIds(iix);
    end

    if isfield(D, 'rf_x')
        cids = unique(D.spikeIds);
        NC = numel(cids);
        D.units = repmat(struct('id', [], 'srf', nan, 'xax', nan, 'yax', nan, 'contour', nan, 'area', nan, 'maxV', nan, 'center', nan), 1, NC);
        th = linspace(0, 2*pi, 100);
        for cc = 1:NC
            D.units(cc).id = cids(cc);
            r = D.rf_r(cids(cc));
            ctr = [D.rf_x(cids(cc)), D.rf_y(cids(cc))];

            D.units(cc).contour = [r*cos(th(:)) + ctr(1), r*sin(th(:)) + ctr(2)];
            D.units(cc).center = ctr;
            D.units(cc).area = pi*r^2;
            D.units(cc).maxV = (D.rf_p(cids(cc)) < 0.05) .* (1- D.rf_p(cids(cc))) .* 100;
        end

    end

    if min(size(D.frameTimes)) > 1
        D.frameTimes = D.frameTimes(:);
        D.framePhase = D.framePhase(:);
        D.frameContrast = D.frameContrast(:);
        [D.frameTimes, ind] = sort(D.frameTimes);
        D.framePhase = D.framePhase(ind);
        D.frameContrast = D.frameContrast(ind);
    end
    
    if all(isnan(D.eyePos(:)))
        D.eyePos = nan(numel(D.frameTimes), 3);
        D.eyeTime = D.frameTimes;
        D.eyeLabels = ones(numel(D.frameTimes),1);
    end

    if any(cellfun(@(x) contains(flist(iSess).name, x), unique_sessions))
        
        unit_offset = max(unique(Dbig.spikeIds));
        if isempty(unit_offset)
            unit_offset = 0;
        end
%         fprintf('offsetting spike ID by %d\n', unit_offset)
        D.spikeIds = D.spikeIds + unit_offset;
        if isfield(D, 'units')
            for ii = 1:numel(D.units)
                D.units(ii).id = D.units(ii).id + unit_offset;
            end
        end
    end
    
    
    
    D.sessNumSpikes = iSess*ones(size(D.spikeTimes));
    D.sessNumGratings = iSess*ones(size(D.GratingOnsets));
    D.sessNumTread = iSess*ones(size(D.treadTime));
    D.sessNumEye = iSess*ones(size(D.eyeTime));

    
    sessStart = 0;
    for iField = 1:numel(timingFields)
        tmp = min(sessStart, min(reshape(D.(timingFields{iField}), [], 1)));
        if isempty(tmp)
            continue
        end
    end
    
    % loop over timing fields and offset time
    for iField = 1:numel(timingFields)
        Dbig.(timingFields{iField}) = [Dbig.(timingFields{iField}); D.(timingFields{iField}) - sessStart + startTime];
        newOffset = max(newOffset, max(Dbig.(timingFields{iField}))); % track the end of this session
    end
    
    for iField = 1:numel(nonTimingFields)
        tmp = D.(nonTimingFields{iField});
        Dbig.(nonTimingFields{iField}) = [Dbig.(nonTimingFields{iField}); tmp];
    end
    
    startTime = newOffset + 2; % 2 seconds between sessions
    if isinf(startTime)
        keyboard
    end
    
    if isfield(D, 'units')
        
        cids = unique(D.spikeIds);
        NC = numel(cids);
        assert(NC==numel(D.units), 'import_supersession: mismatch in number of units on session')
        for cc = 1:NC
            if isfield(D.units(cc), 'area') && ~isnan(D.units(cc).area)
                if numel(Dbig.units) < D.units(cc).id
                    Dbig.units{cids(cc)} = {D.units(cc)};
                else
                    if isempty(Dbig.units{D.units(cc).id})
                        Dbig.units{cids(cc)} = {D.units(cc)};
                    else
                        Dbig.units{cids(cc)} = [Dbig.units{D.units(cc).id} {D.units(cc)}];
                    end
                end
            end
        end
    end

    fprintf('StartTime: %02.2f\n', startTime)
end


%% Cleanup treadmill speed

iix = Dbig.treadSpeed > 200;
treadSpeed = Dbig.treadSpeed;
treadSpeed(iix) = nan;
iix = diff(treadSpeed).^2 > 50;
treadSpeed(iix) = nan;
% treadSpeed = repnan(treadSpeed, 'pchip'); % interpolate between artifacts
Dbig.treadSpeed = treadSpeed;


%% Split supersessioned units if necessary
cids = unique(Dbig.spikeIds);
NC = numel(cids);

if ~strcmp(subj, 'allen') % allen institute data has no super sessions
    for cc = 1:NC
        fprintf('%d/%d\n', cc, NC)
        cids = unique(Dbig.spikeIds);
        cid = cids(cc);

        unitix = Dbig.spikeIds == cid;
        sessix = unique(Dbig.sessNumSpikes(unitix)); % session ids for this unit

        if numel(sessix) == 1 % not supersessioned
            continue
        end

        % get grating onsets
        gtix = find(ismember(Dbig.sessNumGratings, sessix));
        gtix(isnan(Dbig.GratingDirections(gtix))) = [];

        onsets = Dbig.GratingOnsets(gtix);
        winsize = mode(Dbig.GratingOffsets(gtix) - Dbig.GratingOnsets(gtix));

        % count spikes while gratings are on
        t0 = min(onsets) - 2*winsize;
        st = Dbig.spikeTimes(unitix) - t0;
        onsets = onsets - t0;

        st(st < min(onsets)) = [];
        sp = struct('st', st, 'clu', ones(numel(st),1));

        R = binNeuronSpikeTimesFast(sp, onsets, winsize);
        R = R ./ winsize; % spike rate


        figure(1); clf
        subplot(2,1,1)
        plot(R, 'k'); hold on
        for i = sessix(:)'
            inds = find(Dbig.sessNumGratings(gtix)==i);
            plot(inds, R(inds), '.')
        end

        % do an ANOVA to see if their is an
        [pval,~,STATS] = anova1(R, Dbig.sessNumGratings(gtix), 'off');
        if pval < 0.001 % sessions need to be split
            % do all pairwise comparisons
            cmat = multcompare(STATS, 'display','off');
            npairs = size(cmat,1);
            meanrat  = zeros(npairs,1);
            for ipair = 1:npairs
                meanrat(ipair) = max(STATS.means(cmat(ipair,1:2))) ./ min(STATS.means(cmat(ipair,1:2)));
            end

            pairs2combine = cmat(cmat(:,end)>1e-6 | meanrat < 1.2, 1:2); % combine any session pairs that are not highly significantly different
            % identify units that will get split
            units2split = setdiff(unique(cmat(:,1:2)),unique(pairs2combine));

            % we need to do some bookkeeping to make unit ids equal the group
            % nums from the anova

            oclu = ones(1, numel(Dbig.spikeIds(unitix))); % old cluster id
            sessi = Dbig.sessNumSpikes(unitix); % old session id
            sessids = unique(sessi); % session included for this unit

            % build a map from the old session id to the group num in the anova
            sessnum = 1:numel(sessids);
            sessmap = zeros(max(sessi),1);
            sessmap(sessids) = sessnum;
            nsessi = sessmap(sessi); % session ids mapped to group num

            % all units will be added to the end of the unit numbers
            cidoffset = max(cids);

            nclu = oclu; % new unit ids
            fprintf('%d sessions in original unit\n', numel(sessids))

            if ~isempty(units2split)
                fprintf('Splitting ')
                fprintf('%d ', units2split)
                fprintf('\n')

                % split off the sessions that need splitting
                for iiu = units2split(:)'
                    nclu(nsessi == iiu) = cidoffset + iiu;

                    subplot(2,1,2)
                    inds = find(Dbig.sessNumGratings(gtix)==sessids(iiu));
                    plot(inds, R(inds), '.'); hold on
                end
            end

            % combine the sessions that should stay combiend
            units2combine = unique(pairs2combine(:));
            while ~isempty(units2combine)
                if isempty(pairs2combine)
                    combining = units2combine;
                else
                    combining = unique(pairs2combine(any(pairs2combine == units2combine(1),2),:));
                    if isempty(combining)
                        combining = unique(pairs2combine(any(pairs2combine == units2combine(2),2),:));
                    end

                    ncomb = numel(combining);
                    while true
                        combining = unique(pairs2combine(any(ismember(pairs2combine, combining),2),:));
                        if numel(combining) == ncomb
                            break
                        else
                            ncomb = numel(combining);
                        end
                    end
                end

                if any(diff(combining) > 1)
                    d = diff(combining)==1;
                    if numel(d)==1
                        combining(end) = [];
                    else
                        bw = bwlabel(d);
                        bw = [1; bw];
                        n = arrayfun(@(x) sum(bw==x), 1:max(bw));
                        [~, mxid] = max(n);
                        combining = combining(bw==mxid);
                    end
                end
                fprintf('Combining ')
                fprintf('%d ', combining)

                inds = find(ismember(Dbig.sessNumGratings(gtix), sessids(combining)));
                subplot(2,1,2)
                plot(inds, R(inds), '.'); hold on

                %             if numel(combining) < 2
                % %                 units2combine = [];
                %                 break
                %             end
                newid = cidoffset + min(combining);
                fprintf('into %d\n', newid)
                if ~isempty(Dbig.units) && ~isempty(Dbig.units{cid})
                    if numel(Dbig.units) < newid
                        Dbig.units{newid} = {};
                    end

                    existingmaps = numel(Dbig.units{cid});
                    try
                        Dbig.units{newid} = [Dbig.units{newid} Dbig.units{cid}(ismember(existingmaps, combining))];
                    end
                end

                nclu(ismember(nsessi, combining)) = newid;
                units2combine(ismember(units2combine, combining)) = [];
                pairs2combine(any(ismember(pairs2combine, combining), 2),:) = [];
                %             pairs2combine = pairs2combine(any(ismember(pairs2combine, units2combine),2),:);
            end

            % update the original struct
            Dbig.spikeIds(unitix) = nclu;



        end
        drawnow
        %     keyboard


    end
end

cids = unique(Dbig.spikeIds);
NCnew = numel(cids);
fprintf('Number of units increased from %d to %d\n', NC, NCnew)

%% combine RFs
hasrf = find(~cellfun(@isempty, Dbig.units));

NRF = numel(hasrf);

for cc = 1:NRF
    fprintf('RF Cleanup: %d/%d\n', cc, NRF)

    cid = hasrf(cc);

    N = numel(Dbig.units{cid});

    if N > 1
        xc = cellfun(@(x) x.center(1), Dbig.units{cid}(:));
        yc = cellfun(@(x) x.center(2), Dbig.units{cid}(:));

        d = [xc yc];
        z = linkage(d, 'ward', 'euclidean');
        c = cluster(z,'Cutoff', 1.5);

        if numel(unique(c)) > 1
            error('import_supersession: RFs moved for the SS unit. Something is wrong. Probably should split unit.')
        else
            srf = 0;
            for i = 1:N
                %     subplot(sx, sy, i)
                %     imagesc(D.units{cid}{i}.xax, D.units{cid}{i}.yax, D.units{cid}{i}.srf); axis xy
                srf = srf + Dbig.units{cid}{i}.srf;
            end
            srf = srf / N;

            xax = Dbig.units{cid}{i}.xax;
            yax = Dbig.units{cid}{i}.yax;
            [xi, yi] = meshgrid(xax, yax);

            [con, ar, ctr] = get_rf_contour(xi, yi, srf);
            Dbig.units{cid}(2:end) = [];
            Dbig.units{cid}{1}.srf = srf;
            Dbig.units{cid}{1}.area = ar;
            Dbig.units{cid}{1}.center = ctr;
            Dbig.units{cid}{1}.contour = con;
        end
    end

end

% %% Cleanup low firing rate units
% cids = unique(Dbig.spikeIds);
% NC = numel(cids);
% 
% for cc = 1:NC
%     st = Dbig.spikeTimes(Dbig.spikeIds==cids(cc));
%     gtix = Dbig.GratingOnsets > min(st) & Dbig.GratingOffsets < max(st);
% 
%     spix = getTimeIdx(st, Dbig.GratingOnsets(gtix), Dbig.GratingOffsets(gtix));
%     spikerate = sum(spix) / sum(Dbig.GratingOffsets(gtix)-Dbig.GratingOnsets(gtix));
% 
%     if spikerate < 1
%         fprintf('Removing unit %d for low spikes\n', cids(cc))
%         iix = Dbig.spikeIds==cids(cc);
%         Dbig.spikeTimes(iix) = [];
%         Dbig.spikeIds(iix) = [];
%         Dbig.sessNumSpikes(iix) = [];
%         if ~isempty(Dbig.units)
%             Dbig.units{cids(cc)} = [];
%         end
%     end
% end
% 
% cids = unique(Dbig.spikeIds);
% NCnew = numel(cids);
% fprintf('Number of Units after removal went from %d to %d\n', NC, NCnew)

%%

fprintf('Saving... ')
save(fullfile(fpath, [subj 'D_all.mat']), '-v7.3', '-struct', 'Dbig')
disp('Done')



