
%%
freg = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'regression');
fgrat = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'gratings');

flist = dir(fullfile(freg, '*.mat'));

%%

ifile = 31;
D = load(fullfile(fgrat, flist(ifile).name))
%%
ifile = ifile + 1;

load(fullfile(freg, flist(ifile).name))


figure(1); clf

% Show model comparison
models = fieldnames(Rpred);
nModels = numel(models);
model_pairs = nchoosek(1:nModels, 2);
npairs = size(model_pairs,1);

% first get the xlims for the comparison
m = 0;
for i = 1:npairs
    m = max(max(abs(Rpred.(models{model_pairs(i,1)}).Rsquared- Rpred.(models{model_pairs(i,2)}).Rsquared)),0);
end

inc_thresh = .01;
figure(1); clf
for i = 1:npairs
    subplot(2,npairs, i)
    
    m1 = Rpred.(models{model_pairs(i,1)}).Rsquared;
    m1name = models{model_pairs(i,1)};
    m2 = Rpred.(models{model_pairs(i,2)}).Rsquared;
    m2name = models{model_pairs(i,2)};
    
    if mean(m1) > mean(m2)
        mtmp = m1;
        mtmpname = m1name;
        m1 = m2;
        m1name = m2name;
        m2 = mtmp;
        m2name = mtmpname;
    end
    
    plot(m1, m2, 'ow', 'MarkerSize', 6, 'MarkerFaceColor', repmat(.5, 1, 3)); hold on
    plot(xlim, xlim, 'k')
    plot([0 inc_thresh], [inc_thresh inc_thresh], 'k--')
    plot([inc_thresh inc_thresh], [0 inc_thresh], 'k--')
    xlabel(m1name)
    ylabel(m2name)
    
    subplot(2,npairs,i+npairs)
    iix = m2 > inc_thresh;
    histogram(m2(iix) - m1(iix), ceil(sum(iix)/2), 'FaceColor', repmat(.5, 1, 3))
    xlabel(sprintf('%s - %s', m2name, m1name))
    xlim([-m*.5 m])
end

% two model comparison

model1 = 'stimsac';
model2 = 'stimrunsac';

m1 = Rpred.(model1).Rsquared;
m2 = Rpred.(model2).Rsquared;

figure(1); clf
cmap = lines;
plot(m1, 'o', 'MarkerFaceColor', cmap(1,:))
hold on
plot(m2, 'o', 'MarkerFaceColor', cmap(1,:))

legend({model1, model2})

ylabel('Var Explained')
xlabel('Neuron')


ecc = opts.rf_eccentricity;
figure(3); clf
ecc_ctrs = [1 20];
[~, id] = min(abs(ecc(:) - ecc_ctrs), [], 2);
cmap = lines;
clear h
for i = unique(id)'
    h(i) = plot(m1(id==i), m2(id==i), 'ow', 'MarkerFaceColor', cmap(i,:)); hold on
    plot(xlim, xlim, 'k')
    xlabel(sprintf('Model [%s] r^2', model1))
    ylabel(sprintf('Model [%s] r^2', model2))
end
legend(h, {'foveal', 'peripheral'})
title(flist(ifile).name)