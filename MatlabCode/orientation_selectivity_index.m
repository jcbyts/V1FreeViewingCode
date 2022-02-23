function DSI = orientation_selectivity_index(orientations, mu, circ)
% DSI = direction_selectivity_index(directions, mu, circular)
if nargin < 3
    circ = false;
end

if circ == 2
    DSI = sqrt(  (mu(:)'*sin(orientations(:)*2)).^2 + (mu(:)'*cos(orientations(:)*2)).^2) / sum(mu);
elseif circ==1
    DSI = abs(sum(mu .* exp(2*1i*(orientations/180*pi)))) ./ sum(mu);
else
    circdiff = @(th1, th2) angle(exp(1i*(th1-th2)/180*pi))/pi*180;
    [Rpref, id] = max(mu);
    [~, nullid] = min(abs(circdiff(mod(orientations(id)+90, 180), orientations)));
    Rnull = mu(nullid);
    DSI = (Rpref - Rnull) ./ (Rpref + Rnull);
end