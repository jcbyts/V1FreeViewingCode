function [slist,removeCounter] = remove_double_saccades(slist, smo, plotit)
% slist = remove_double_saccades(slist, smo, plotit)

if nargin < 3
    plotit = false;
end

SOFTTHRESH = 0.2;
HARDTHRESH = 0.03; % 20ms

spd = smo(:,7);
pos = hypot(smo(:,2), smo(:,3));

nsac = size(slist,1);
isac = 1;
removeCounter = 0;

while isac < nsac
    
    isac = isac + 1;
    
    while (slist(isac,1) - slist(isac-1,2)) < HARDTHRESH
        slist(isac,:) = [];
        nsac = size(slist, 1);
        removeCounter = removeCounter + 1;
    end
    
    % fit gaussians to the two saccades seperately and combined
    r2 = 1;
    r22 = 0;
    
    d = (slist(isac,1) - slist(isac-1,2));
    
    while (r2 > r22) && (d < SOFTTHRESH)
        remove = 0;
%         d = (slist(isac,1) - slist(isac-1,2));
        
        % fir one gaussian to both saccades
        x = slist(isac-1,4):slist(isac,5);
        y = spd(x)';
    
        [phat, fun] = fit_gaussian_poly(x,y);
        yhat = fun(phat, x);
    
        % fit both separately
        x1 = slist(isac-1,4):slist(isac-1,5);
        y1 = spd(x1)';
    
        [phat, fun] = fit_gaussian_poly(x1,y1);
        yhat1 = fun(phat, x);
    
        x2 = slist(isac,4):slist(isac,5);
        y2 = spd(x2)';
    
        [phat, fun] = fit_gaussian_poly(x2,y2);
        phat(end) = 0; % no offset for this one
        yhat2 = fun(phat, x);
    
        r2 = rsquared(yhat,y);
        r22 = rsquared(yhat1+yhat2, y);
        r23 = rsquared(yhat1, y);
        
        
        r2(isnan(r2)) = -inf;
        r22(isnan(r22)) = -inf;
        
        px = pos(x);
        px1 = pos(x1);
        px2 = pos(x2);
    
        if r23 > r22
            remove = 3;
        end
            
        % single fit better than double fit --> combine into previous
        % saccade and remove
        if (r2 > r22) && (r2 > 0)
            if (x(end) - x(1)) < SOFTTHRESH
                remove = 2;
            else
                remove = 1;
            end
        end
        
        % no speed bump --> remove, this isn't a saccade
        if (r2 < 0) && (r22 < 0)
            remove = 1;
        end
        
            
        if plotit
            figure(1); clf
            subplot(2,1,1)
            plot(x, px, 'k'); hold on
            plot(x1,px1, '.');
            plot(x2,px2, '.')
            
            subplot(2,1,2)
            x = x - x(1);
            x = x/1e3;
            plot(x,y); hold on
            plot(x, yhat)
            plot(x, yhat1+yhat2)
            title([remove r2 r22 r23])
            
            pause
            
        end
         
        % do removal
        switch remove
            case 1
                slist(isac,:) = [];
                nsac = size(slist, 1);
                removeCounter = removeCounter + 1;
                
            case 2
                slist(isac-1,2) = slist(isac,2);
                slist(isac-1,3) = nan;
                slist(isac-1,5) = slist(isac,5);
                slist(isac-1,6) = nan;
                slist(isac-1,7) = 7;
                slist(isac,:) = [];
                nsac = size(slist, 1);
                removeCounter = removeCounter + 1;
            case 3
                slist(isac,:) = [];
                nsac = size(slist, 1);
                removeCounter = removeCounter + 1;
        end
        
        d = (slist(isac,1) - slist(isac-1,2));
    
    end
    
end
    
    
% find nans and fill in the peak velocity
needsPeak = find(isnan(slist(:,3)));
for sac = needsPeak(:)'
    
    iix = slist(sac,4):slist(sac,5);
    [~, id] = max(spd(iix));
    slist(sac,3)=smo(iix(id));
    slist(sac,6)=iix(id);
end

    
   


