
%%
freg = '/home/jake/Data/Datasets/HuklabTreadmill/regression/';

flist = dir(fullfile(freg, '*.mat'));
ifile = 0;

subjs = {'allen', 'gru', 'brie'};
Reg = struct();
Dsum = struct();

models2test = {'drift', 'stimsac', 'stimrunsac', 'RunningGain', 'PupilGain'};


for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    sublist = find(arrayfun(@(x) contains(x.name, subj), flist))';
    
    Reg.(subj) = struct();
    Dsum.(subj) = struct('ntrials', []);

    for imodel = 1:numel(models2test)
        Reg.(subj).(models2test{imodel}).rsquared = [];
        Reg.(subj).(models2test{imodel}).cc = [];
    end

    for ifile = sublist(:)'
        fprintf('Loading [%s]\n', flist(ifile).name)
        load(fullfile(freg, flist(ifile).name))
        Dsum.(subj).ntrials = [Dsum.(subj).ntrials; numel(Stim.grating_onsets)];
 
        for imodel = 1:numel(models2test)
            Reg.(subj).(models2test{imodel}).rsquared = [Reg.(subj).(models2test{imodel}).rsquared; Rpred_ind.(models2test{imodel}).Rsquared(:)];
            Reg.(subj).(models2test{imodel}).cc = [Reg.(subj).(models2test{imodel}).cc; Rpred_ind.(models2test{imodel}).CC(:)];
        end
    end
end
%%
figure(1); clf;
for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    histogram(Dsum.(subj).ntrials, 'binEdges', linspace(0, 1300, 20)); hold on
end
legend(subjs)
xlabel('Number of trials')
ylabel('Number of sessions')
set(gcf, 'Color', 'w')
%% cells worth analyzing with the stimulus
for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    iix = Reg.(subj).stimsac.rsquared > Reg.(subj).drift.rsquared;
    iix = iix & Reg.(subj).stimsac.rsquared > 0;
    fprintf('%s has %d / %d good units\n', subj, sum(iix), numel(iix))
end


% subjs = {'allen', 'gru', 'brie'};
subjs = {'allen'};

figure(2); clf

set(gcf, 'Color', 'w')
for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    histogram(Reg.(subj).stimsac.rsquared, 'binEdges', linspace(-1, 1, 200)); hold on
end
legend(subjs)
xlabel('cv r^2')

%% plot subject dependent responses
mbase = 'PupilGain';
mtest = 'RunningGain';
% subjs = {'brie'};

figure; clf

clear h
for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    
    iix = Reg.(subj).stimsac.rsquared > Reg.(subj).drift.rsquared;
%     iix = iix & Reg.(subj).stimsac.rsquared > 0;

    r2base = Reg.(subj).(mbase).rsquared(iix);
    r2test = Reg.(subj).(mtest).rsquared(iix);
    r2base = max(r2base, -.1);
    r2test = max(r2test, -.1);

    h(isubj) = plot(r2base, r2test, '.'); hold on

end



plot(xlim, xlim, 'k')
xlabel(mbase)
ylabel(mtest)
% title(subj)
legend(h, subjs)

plot.fixfigure(gcf, 12, [5 5])


%% look at individual session
ifile = ifile + 1;

if ifile > numel(flist)
    ifile = 1;
end

fprintf('Loading [%s]\n', flist(ifile).name)
load(fullfile(freg, flist(ifile).name))

exname = strrep(strrep(flist(ifile).name, '_', ' '), '.mat', '');

mbase = 'stimsac';
mtest = 'stimrunsac';

r2base = Rpred_ind.(mbase).Rsquared;
r2test = Rpred_ind.(mtest).Rsquared;

figure(1); clf
plot(r2base, r2test, '.'); hold on
plot(xlim, xlim, 'k')
xlabel(mbase)
ylabel(mtest)
title(exname)

iix = max(r2test, r2base) > 0.05;
% clf
% plot(r2test(iix) ./ r2base(iix), '.'); hold on
% plot(xlim, [1 1], 'k')

rrat = r2test ./ r2base;
sigix = find(rrat > 1.05 & iix);
cc = 0;

figure(2);
plot(sum(Rpred_ind.data.indices.*Rpred_ind.data.Robs), r2test, '.'); hold on
%%
clf
cc = cc + 1;
if cc > numel(sigix)
    cc = 1;
end
R = Rpred_ind.data.Robs(:,sigix(cc));
tread_speed = Stim.tread_speed(:);

good_inds = find(Rpred_ind.data.indices(:,sigix(cc)));
tread_speed = tread_speed(good_inds);
% plot(R(good_inds), 'k'); hold on
plot(imgaussfilt(R(good_inds), 11), 'k'); hold on
plot(tread_speed./max(tread_speed)*max(R), 'c')
hold on


% plot()


%%

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

%% two model comparison

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