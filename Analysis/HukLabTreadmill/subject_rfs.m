

%%
fdir =  '~/Google Drive/HuklabTreadmill/gratings/';
subj = 'brie';

import_supersession(subj, fdir)

%%
subj = 'brie';
D = load_subject(subj, fdir);


%%
hasrf = find(~cellfun(@isempty, D.units));
NC = numel(unique(D.spikeIds));

NRF = numel(hasrf);
figure(3); clf
nsx = ceil(sqrt(NRF));
nsy = round(sqrt(NRF));
ax = plot.tight_subplot(nsx, nsy, .01, 0.01);

for cc = 1:NRF

cid = hasrf(cc);

N = numel(D.units{cid});
% sx = ceil(sqrt(N));
% sy = round(sqrt(N));

% figure(1); clf
srf = 0;
for i = 1:N
%     subplot(sx, sy, i)
%     imagesc(D.units{cid}{i}.xax, D.units{cid}{i}.yax, D.units{cid}{i}.srf); axis xy
    srf = srf + D.units{cid}{i}.srf;
end


xax = D.units{cid}{i}.xax;
yax = D.units{cid}{i}.yax;
srf = srf / N;

figure(3)
set(gcf, 'currentaxes', ax(cc))
srf = srf / std(srf(:));
srf = srf - mean(srf(:));

imagesc(xax, yax, srf, [-5 5])
colormap(plot.coolwarm)
axis xy

% figure(2); clf
% for i = 1:N
%     plot(D.units{cid}{i}.contour(:,1), D.units{cid}{i}.contour(:,2)); hold on
% end

% drawnow
% pause(0.1)
end

