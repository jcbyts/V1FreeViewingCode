function eyePos = getCorrectedEyePosUseProbe(Exp, varargin)
% eyePos = getCorrectedEyePos(Exp, varargin)
% 'usebilinear', false

ip = inputParser();
ip.addParameter('usebilinear', true) % not bilinear, a polynomial, should fix
ip.parse(varargin{:})

validTrials = io.getValidTrials(Exp, 'Forage');

% eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
% figure(1); clf

%%
counter = 0;

vtt = Exp.vpx.smo(:,1);
vtt = Exp.vpx2ephys(vtt);
eyePos = io.getCorrectedEyePos(Exp, 'usebilinear', true);

x = eyePos(:,1);
y = eyePos(:,2);

EXY = [];
PXY = [];

for iTrial = 1:numel(validTrials)
    
    thisTrial = validTrials(iTrial);
    
    tstart = Exp.ptb2Ephys(Exp.D{thisTrial}.STARTCLOCKTIME);
    tend = Exp.ptb2Ephys(Exp.D{thisTrial}.ENDCLOCKTIME);
    
    vpxix = vtt >= tstart & vtt < tend;
    if sum(vpxix) == 0
        continue
    end
    %%
    
%     figure(1); clf
%     plot(x(vpxix), y(vpxix), '.'); hold on
    
    ex = x(vpxix);
    ey = y(vpxix);
    pt = Exp.ptb2Ephys(Exp.D{thisTrial}.PR.ProbeHistory(:,4));
    vt = vtt(vpxix);
    veye = ceil((pt - vt(1)) / median(diff(vt)));
    
    px = Exp.D{thisTrial}.PR.ProbeHistory(:,1);
    py = Exp.D{thisTrial}.PR.ProbeHistory(:,2);
    
    exx = ex(veye);
    eyy = ey(veye);
    d = hypot(exx-px, eyy-py);
    
    fixations = (d < .5);
    pxy = [px(fixations) py(fixations)];
    exy = [exx(fixations) eyy(fixations)];
    
    
    EXY = [EXY; exy];
    PXY = [PXY; pxy];
%     %%
%     tptb = Exp.D{thisTrial}.rewardtimes;
%     t = Exp.ptb2Ephys(tptb);
%     
%     for ii = 1:numel(tptb)
%         idx = find(Exp.D{thisTrial}.PR.ProbeHistory(:,4) < tptb(ii), 1, 'last');
%         eyeidx = find(eyeTime < t(ii), 1, 'last');
%         
%         counter = counter + 1;
%         probeX(counter) = Exp.D{thisTrial}.PR.ProbeHistory(idx-1,1);
%         probeY(counter) = Exp.D{thisTrial}.PR.ProbeHistory(idx-1,2);
%         
%         iix = (-50:0)+eyeidx;
%         eyePos = Exp.vpx.smo(iix, 2:3);
%         bw = bwlabel(Exp.vpx.Labels(iix)==1);
%         
%         eyeX(counter) = mean(eyePos(bw == max(bw),1));
%         eyeY(counter) = mean(eyePos(bw == max(bw),2));
%         
%         subplot(1,2,1)
%         plot(probeX(counter), eyeX(counter), 'ob')
%         hold on
%         subplot(1,2,2)
%         plot(probeY(counter), eyeY(counter), 'ob')
%         hold on
%         
%     end
    
end

figure(1); clf
subplot(1,2,1)
plot(EXY(:,1), PXY(:,1), '.'); hold on
plot(xlim, xlim, 'k')
subplot(1,2,2)
plot(EXY(:,2), PXY(:,2), '.'); hold on
plot(xlim, xlim, 'k')


counter = size(EXY,1);

if ip.Results.usebilinear
    X = [EXY EXY.^2 ones(counter,1)];
else
    X = [EXY ones(counter,1)];
end

Y = PXY;

bad = isnan(sum(X,2)) | isnan(sum(Y,2));
X(bad,:) = [];
Y(bad,:) = [];

w = (X'*X) \ (X'*Y);

% opts = statset('nlinfit');
% opts.RobustWgtFun = 'bisquare';

% Fit the nonlinear model using the robust fitting options. Here, use an expression to specify the model.

% b0 = w(:,1);
% modelstr = 'y ~ b1 + b2*x + b3*x^2';
% 
% mdl = fitnlm(x,y,modelstr,b0,'Options',opts);

xhat = X*w;
subplot(1,2,1)
plot(EXY(:,1), xhat(:,1), '.')
subplot(1,2,2)
plot(EXY(:,2), xhat(:,2), '.')

if ip.Results.usebilinear
    Xeye = [eyePos eyePos.^2 eyePos.^3 ones(size(Exp.vpx.smo,1),1)];
else
    Xeye = [eyePos ones(size(Exp.vpx.smo,1),1)];
end

eyePos = Xeye*w;

figure(1); clf
plot(eyePos(:,1)); hold on
plot(Exp.vpx.smo(:,2))