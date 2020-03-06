

[xx,yy,zz]=ndgrid(1:10, 1:10, 1:10);

F = xx.^2 + yy.^2 + zz.^2;
figure(1); clf
h = scatter3(xx(:),yy(:),zz(:), 's', 'MarkerEdgeColor','k', 'MarkerFaceColor', 'flat' );
h.CData=F(:);
h.SizeData = 2e3;

trisurf(xx(:), yy(:), zz(:), F(:))

%%
figure(1); clf
k = boundary(xx(:), yy(:), zz(:));
h = trisurf(k, xx(:), yy(:), zz(:), F(:));