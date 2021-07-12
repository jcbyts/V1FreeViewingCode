

D = load_subject(subj);

%%
clear Dstat
for sessionId = 1:Nsess
    Dstat(sessionId) = do_decoding(D, sessionId);
end


%% decoding error
circdiff = @(x,y) angle(exp(1i*(x - y)/180*pi))/pi*180;

mR = zeros(Nsess,1);
mRCi = zeros(Nsess,2);
mS = zeros(Nsess,1);
mSCi = zeros(Nsess,2);
nS = zeros(Nsess,1);
nR = zeros(Nsess,1);

figure(1); clf
for iSess = 1:Nsess
    aerr = abs(circdiff(Dstat(iSess).Stim, Dstat(iSess).decoderStimTot));
    
    inds = Dstat(iSess).runTrials;
    nR(iSess) = numel(inds);
    mR(iSess) = median(aerr(inds));
    mRCi(iSess,:) = bootci(1000, @median, aerr(inds));
    
    inds = setdiff(1:Dstat(iSess).NTrials, Dstat(iSess).runTrials);
    nS(iSess) = numel(inds);
    mS(iSess) = median(aerr(inds));
    mSCi(iSess,:) = bootci(1000, @median, aerr(inds));
    
    plot(mS(iSess)*[1 1], mRCi(iSess,:), 'Color', .5*[1 1 1]); hold on
    plot(mSCi(iSess,:), mR(iSess)*[1 1], 'Color', .5*[1 1 1]);
    h = plot(mS(iSess), mR(iSess), 'o')
    h.MarkerFaceColor = h.Color;
    
end

plot(xlim, xlim, 'k')
xlim([0 20])
ylim([0 20])
xlabel('Stationary')
ylabel('Running')
title('Median Decoding Error (degrees)')

%%
figure(1); clf
