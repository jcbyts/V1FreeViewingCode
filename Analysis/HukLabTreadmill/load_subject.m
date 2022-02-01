function D = load_subject(subj, fpath)
% load super session file for subject
% D = load_subject(subj)


if ~exist('fpath', 'var')
    fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
end

if nargin < 1
    subj = 'gru';
end

validSubjs = {'gru', 'brie', 'allen'};
assert(ismember(subj,validSubjs), sprintf("import_supersession: subj name %s is not valid", subj))


fname = fullfile(fpath, [subj 'D_all.mat']);
D = load(fname);

sessions = unique(D.sessNumSpikes);
Nsess = numel(sessions);
NCs = zeros(Nsess,1);
fprintf('Loading subject [%s]\n', subj)
fprintf('Found %d unique sessions\n', Nsess)
for i = 1:Nsess
    NC = numel(unique(D.spikeIds(D.sessNumSpikes == sessions(i))));
    dur = median(D.GratingOffsets(D.sessNumGratings == sessions(i)) - D.GratingOnsets(D.sessNumGratings == sessions(i)));
    
    fprintf('%d) %d Units, %d Trials, Stim Duration: %02.2fs \n', sessions(i), NC, sum(D.sessNumGratings == sessions(i)), dur)
    NCs(i) = NC;
end