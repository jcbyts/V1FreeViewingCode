function DSI = direction_selectivity_index(directions, mu, circ, plotit)
% DSI = direction_selectivity_index(directions, mu, circular)

if nargin < 4
    plotit = false;
end

if nargin < 3
    circ = 0;
end

if circ == 2
    DSI = sqrt(  (mu(:)'*sin(directions(:))).^2 + (mu(:)'*cos(directions(:))).^2) / sum(mu);
elseif circ == 1
    DSI = abs(sum(mu .* exp(2*1i*(directions/180*pi)))) ./ sum(mu);
else
    circdiff = @(th1, th2) angle(exp(1i*(th1-th2)/180*pi))/pi*180;
    [Rpref, id] = max(mu);
    [~, nullid] = min(abs(circdiff(mod(directions(id)+180, 360), directions)));
    Rnull = mu(nullid);
    DSI = (Rpref - Rnull) ./ (Rpref + Rnull);
    if plotit
        figure,
        plot(directions, mu, 'k'); hold on
        plot([1 1]*directions(id), ylim, 'b')
        plot([1 1]*directions(nullid), ylim, 'r')
    end
end