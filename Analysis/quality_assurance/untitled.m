%%
CSD = cstruct1.CSD;
time = cstruct1.time;
ch0 = cstruct1.chDepths;

%% find the sink and reversal point
tpower = std(CSD).^4.*-1;
ix = time < 0;
    
% smooth temporally
tpower = sgolayfilt(tpower, 1, 5);
    
tpower = fix(tpower/ (10*max(tpower(ix))));
figure(1); clf
plot(tpower)
hold on
    
dpdt = diff(tpower);
    
inds = find(sign(dpdt)~=0); % remove indices that don't have a sign
    
zc = inds(diff(sign(dpdt(inds)))==-2);

[~, ind] = sort(tpower(zc), 'descend');
    
if length(zc) >= 3
    numP = 3;
else
    numP = length(zc);
end
    
    
zc = sort(zc(ind(1:min(numel(ind), 3))))+1;

plot(zc, tpower(zc), 'o')

while time(zc(1)) < 20
    zc(1) = [];
end

firstpeak = zc(1)+[-1 0 1];

plot(firstpeak, tpower(firstpeak), 'ok')
%%

spower = mean(CSD(:,firstpeak), 2);
    
figure(1); clf
plot(spower); hold on


% get peak
[~, peaks] = findpeaks(spower, 'MinPeakWidth', 2, 'MinPeakHeight', .5);
[~, vals] = findpeaks(-spower, 'MinPeakWidth', 2, 'MinPeakHeight', .5);
    
% find source
mx = min(peaks);

% look for sink above the source
mn = min(vals(vals>mx));
    
plot(mx, spower(mx), 'o')
plot(mn, spower(mn), 'd')

%%
assert(mn > mx, 'source must be above sink to work')
    
ind = mx:mn;
rvrsl = ind(find(diff(sign(spower(ind)))==-2, 1));

t = time(firstpeak(2));
    
source = ch0(mx+1); % plus 1 because CSD cuts channel (due to derivative)
sink = ch0(mn+1); % same here
reversal = ch0(rvrsl+2); % plus 2 because we took the derivative again to
    
if isempty(source)
    source = nan;
    mx = nan;
end
if isempty(sink)
    sink = nan;
    mn = nan;
end

if isempty(reversal)
    reversal = nan;
end

%
figure(2); clf
imagesc(time, ch0, CSD); hold on
plot(t, sink, 'ow', 'MarkerSize', 20)
plot(t, source, 'ow', 'MarkerSize', 20)
plot(xlim, reversal*[1 1], 'r--')