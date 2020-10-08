function S = detect_doublestep_saccades(Exp, varargin)
% detect doublestep saccades
% S = detect_doublestep_saccades(Exp, varargin)
% Input:
%   Exp (struct): Marmoview struct
%   'thresh' (optional): threshold on fixation duration to count as suspiciously brief
%   'debug' (optional): boolean. stop and plot each saccade pair

ip = inputParser();
ip.addParameter('thresh', 0.05);
ip.addParameter('debug', false);
ip.parse(varargin{:});

saconset = Exp.slist(:,1);
sacoffset = Exp.slist(:,2);

fixdur = saconset(2:end) - sacoffset(1:end-1);

thresh = ip.Results.thresh;
briefFixIdx = find(fixdur < thresh); % fixations less than 50 ms

nFix = numel(briefFixIdx);

S.suspectSaccadePairs = [briefFixIdx(:) briefFixIdx(:)+1];
S.s1dxdy = nan(nFix, 2);
S.s2dxdy = nan(nFix, 2);
S.dotprod = nan(nFix,1);

buffer = 100; % samples

for iFix = 1:nFix
    
    thisFix = briefFixIdx(iFix);
    assert( (saconset(thisFix+1)-sacoffset(thisFix)) < thresh, "indexing bug: this fixation is longer than the threshold")
    
    eixs = Exp.slist(thisFix,4) - buffer;
    eixe = Exp.slist(thisFix+1,5) + buffer;
    
    inds = eixs:eixe;
    
    eyeX = Exp.vpx.smo(inds, 2);
    eyeY = Exp.vpx.smo(inds, 3);
    
    sac1 = Exp.slist(thisFix,4):Exp.slist(thisFix,5);
    sac2 = Exp.slist(thisFix+1,4):Exp.slist(thisFix+1,5);
    
    s1x = eyeX(ismember(inds, sac1));
    s1y = eyeY(ismember(inds, sac1));
    
    s1dx = s1x(end) - s1x(1);
    s1dy = s1y(end) - s1y(1);
    
    s2x = eyeX(ismember(inds, sac2));
    s2y = eyeY(ismember(inds, sac2));
    
    s2dx = s2x(end) - s2x(1);
    s2dy = s2y(end) - s2y(1);
    
    
    u = [s1dx; s1dy];
    v = [s2dx; s2dy];
    
    S.s1dxdy(iFix,:) = u;
    S.s2dxdy(iFix,:) = v;
    
    dp = (u'*v)/norm(u)/norm(v);
    
    S.dotprod(iFix) = dp;
    
    if ip.Results.debug
        figure(1); clf
        subplot(1,2,1)
        plot(eyeX, eyeY); hold on
        plot(s1x, s1y, 'r');
        plot(s2x, s2y, 'g');
        
        quiver(s1x(1), s1y(1), s1dx, s1dy, 'r');
        quiver(s2x(1), s2y(1), s2dx, s2dy, 'g');
        
        subplot(1,2,2)
        plot(inds,eyeX); hold on
        plot(inds,eyeY)
        
        plot(sac1, s1x, 'r')
        plot(sac1, s1y, 'r')
        
        plot(sac2, s2x, 'g')
        plot(sac2, s2y, 'g')
        
        title(dp)
        
        pause
    end
end
