

%%
flist = dir('Figures/2021_pytorchmodeling/*model2.mat');

f = 2;

tmp = load(fullfile(flist(f).folder, flist(f).name));


%%

figure(1); clf
imagesc(tmp.wreadout)



%%

