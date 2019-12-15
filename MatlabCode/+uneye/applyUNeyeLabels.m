function Exp = applyUNeyeLabels(Exp, dataDir)
% saves out csv files for training with UNeye

if nargin < 2
    dataDir = '~/Dropbox/Projects/FreeViewing/Data/uneye';
end

fname = strrep(Exp.FileTag, '.mat', '');

X0 = Exp.vpx.smo(:,2);
L0 = Exp.vpx.Labels;

n = numel(X0);

X = csvread(fullfile(dataDir, [fname 'Xmat.csv']));
L = csvread(fullfile(dataDir, [fname 'Luneye.csv']));

% reshape to column
X2 = reshape(X', [], 1);
L2 = reshape(L', [], 1);

% remove padding
X2 = X2(1:n);
L2 = L2(1:n)+1; % matlab is 1-based

fprintf('New labels are %02.2f%% similar\n', mean(L0 == L2)*100);


% Eliminate labels that are sandwiched between lost track
lost = L2==4;
lstart = find(diff(lost)==1);
lstop = find(diff(lost)==-1);
if lost(1)
    lstart = [1; lstart];
end

if lost(end)
    lstop = [lstop; numel(lost)];
end

lnew = lstart(2:end);
lold = lstop(1:end-1);
fillin = find((lnew-lold) < 15);
for i = 1:numel(fillin)
    L2(lold(fillin(i)):lnew(fillin(i))) = 4;
end

% now detect saccades
saccades = L2 == 2;
sstart = find(diff(saccades)==1);
sstop = find(diff(saccades)==-1);
if saccades(1)
    sstart = [1; sstart];
end

if saccades(end)
    sstop = [sstop; numel(saccades)];
end

% correct for saccades that are only one sample long
bad = sstop((sstop-sstart)==1);
for i = 1:numel(bad)
    L2(bad(i)) = L2(bad(i)+1);
end

% correct for intersaccade intervals that are unreasonable
snew = sstart(2:end);
sold = sstop(1:end-1);

bad = find((snew-sold)<5);
for i = 1:numel(bad)

    % debugging plots
%    b = snew(bad(i));
%    win = -100:100;
%    figure(1); clf
%    subplot(2,1,1)
%    plot(win, Exp.vpx.smo(b + win,2))
%    subplot(2,1,2)
%    plot(win, L2(b+win), 'o-'); hold on
   
   L2(sold(bad(i)):snew(bad(i))) = 2;
%    plot(win, L2(b+win), 'o-');
end

% do it again
saccades = L2 == 2;
sstart = find(diff(saccades)==1);
sstop = find(diff(saccades)==-1);
if saccades(1)
    sstart = [1; sstart];
end

if saccades(end)
    sstop = [sstop; numel(saccades)];
end

bad = sstop((sstop-sstart)==1);
assert(numel(bad)==0)


nSaccades = numel(sstart);
midpoint = zeros(nSaccades, 1);
for iSaccade = 1:nSaccades
    x = sstart(iSaccade):sstop(iSaccade);
    y = Exp.vpx.smo(x,7);
    id = round(sum(x(:).*y(:) ./ sum(y)));
    
    midpoint(iSaccade) = id;
    if isnan(midpoint(iSaccade))
        L2(x) = 4;
    end
        
end

bad = isnan(midpoint);
sstart(bad) = [];
sstop(bad) = [];
midpoint(bad) = [];


figure
plot(X0, 'k'); hold on
cmap = lines;
for i = 1:4
    ix = find(L2==i);
    plot(ix, X2(ix), '.', 'Color', cmap(i,:)); hold on
end

% save it out
sl = [Exp.vpx.smo(sstart,1) Exp.vpx.smo(sstop,1) Exp.vpx.smo(midpoint,1) sstart sstop midpoint];
Exp.slist = sl;
Exp.vpx.Labels = L2;          

save(fullfile(dataDir, Exp.FileTag), '-v7.3', '-struct','Exp')


