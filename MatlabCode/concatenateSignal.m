function xout = concatenateSignal(x, inds)
% xout = concatenateSignal(x, inds)
% concatenate signal at indices
% x @double [1 x n] (the signal)
% inds @cell (cell array of indices to concatenate)

if size(x,1) > size(x,2)
    transpose = true;
    x = x';
end

xout = x(inds{1});
for j = 2:numel(inds)
    tmp = x(inds{j}) - x(inds{j}(1));
    tmp = tmp + xout(end);
    xout = [xout tmp];
end
    
if transpose
    xout = xout';
end