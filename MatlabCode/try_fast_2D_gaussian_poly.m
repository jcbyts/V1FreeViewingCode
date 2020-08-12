
%% Try fast 2D gaussian fitting

dims = [64 64]; % size of the ROI
xax = linspace(-dims(1)/2, dims(1)/2, dims(1));
yax = linspace(-dims(2)/2, dims(2)/2, dims(2));
[xx,yy] = meshgrid(xax, yax);

mask = hypot(xx, yy)>25;
figure(1); clf
imagesc(mask)

X = [xx(:) yy(:)]; % stimulus coordinates (need to precompute for fast computation)

% setup polynomial matrix
n = numel(xx);
Xd = [ones(n,1) X X(:,1).^2 2*X(:,1).*X(:,2) X(:,2).^2];

amp = 150;
noisesigma = 20;
Q = [5^2 5; 5 5^2]; % covariance
mu = [0 0];
thresh = .2; % only use points that have nice logs

nsim = 100;
tclock = nan(nsim, 1); % duration of actual fitting
tclock2 = nan(nsim,1);
mu_true = nan(nsim, 2);
mu_gfit = nan(nsim, 2);
mu_rsc = nan(nsim,2);

plotit = true;


for isim = 1:nsim

% %     mu = mu + randn(1,2)*.2; % random walk
%     s = Q(2) + randn(1)*.02; % change rotation;
%     Q(2:3)=s;
%     Q(1) = max(Q(1) + randn(1)*.5, .5); % must be positive
%     Q(4) = max(Q(4) + randn(1)*.5, .5);
    
    %%
    % tru params
    params = [amp, mu, Q(1:2), Q(4)];


    % make true gaussian
    I = mvnpdf(X, params(2:3), [params(4:5); params(5:6)]);
    I = I/max(I(:))*params(1);
    I = I + randn(size(I))*noisesigma; % add noise
    
   
    I = (min(max(I,0), 255)); % impose bit-depth
    I = imgaussfilt(reshape(I, dims), 2).^4;
    Y = I(:); % observed data
    
    plot(Y(:), '.')
    %%
    if plotit
        figure(1); clf
        imagesc(xax, yax, reshape(Y, size(xx)))
        colormap gray
        axis xy
        hold on
    end

    % gaussian fitting using polynomial trick
    tic
    Ynorm = (Y - min(Y))/ (max(Y) - min(Y)); % it helps to normalize
    Ynorm(mask(:)) = 0;
    use = Ynorm > thresh;

    ly = log(Ynorm(use));

%     phat = Xd(use,:)\ly;
    nu = sum(use);
    
    % compare to radial symmetric center
    I = reshape(Ynorm, size(xx));
    tic
    [x, y] = radialcenter(I);
    x = x - dims(1)/2 -.5 ;
    y = y - dims(2)/2 -.5 ;
    tclock2(isim) = toc;
    
    mu_rsc(isim,:) = [x y];
    
    
    % iter 0
    % initial guess
    Q0 = [10 0; 0 10];
    mu0 = [x,y];
    
    plot.plotellipse(mu0, Q0, 1); hold on
    
    phat = zeros(6,1);
    phat(2:3) = Q0\mu0';
    
    Q0 = -inv(Q0)/2;
    phat(4:6) = Q0([1 2 4]);
    phat(1) = max(ly);
    wd = 1 ./ max(eps, abs(ly-Xd(use,:)*phat));
    W = diag(wd/max(wd));
    
    figure(2); clf
    for iter = 1
        figure(2)
        plot3(X(use, 1), X(use,2), (Xd(use,:)*phat), 'o'); hold on
        plot3(X(use, 1), X(use,2),ly, '.');
        pause
%         
        figure(1)
        phat = (Xd(use,:)'*W*Xd(use,:) + 1e-6*eye(6))\(Xd(use,:)'*W*ly);
        % iter 1
        wd = 1 ./ max(eps, abs(ly-Xd(use,:)*phat));
        W = diag(wd);
%     
%     phat = (Xd(use,:)'*W*Xd(use,:) + 1e-6*eye(6))\(Xd(use,:)'*W*ly);
%     
%     % iter 2
%     wd = 1 ./ max(1e-5, abs(ly-Xd(use,:)*phat));
%     W = diag(wd);
%     phat = (Xd(use,:)'*W*Xd(use,:) + 1e-6*eye(6))\(Xd(use,:)'*W*ly);
%     
    
    QHat = -inv(2*phat([4:5; 5:6]));
    
    muHat = QHat*phat(2:3);

    tclock(isim) = toc; % store time to fit
    
    % store performance
    mu_true(isim,:) = mu;
    mu_gfit(isim,:) = muHat;

    
    
    if plotit
        plot.plotellipse(muHat, QHat, 1); hold on
        plot(muHat(1), muHat(2), 'og')
        plot(x,y, 'or')
        plot(0,0,'+r')
    end
    end

end

%% plot it
figure(1); clf
subplot(2,1,1)
plot(mu_true(:,1), 'k'); hold on
plot(mu_gfit(:,1), 'g')
plot(mu_rsc(:,1), 'r')
ylabel('x')

subplot(2,1,2)
plot(mu_true(:,2), 'k'); hold on
plot(mu_gfit(:,2), 'g')
plot(mu_rsc(:,2), 'r')
ylabel('y')
legend({'data', 'gaussfit', 'radial center'})

%% how bad was it
squerrorG = sum((mu_true-mu_gfit).^2,2);
squerrorR = sum((mu_true-mu_rsc).^2,2);
figure(2); clf
histogram(squerrorG, 50); hold on
histogram(squerrorR, 50)

mseG = mean(squerrorG);
mseRSM = mean(squerrorR);
fprintf('MSE for Gpoly %02.3f, RSC %02.3f\n', mseG, mseRSM)

%% timing
figure(1); clf
histogram(tclock*1e3, 'binEdges', linspace(0, 1, 100)); hold on
histogram(tclock2*1e3, 'binEdges', linspace(0, 1, 100))
xlabel('ms')
legend({'Gaussian fit', 'radial symmetric center'})

%% visualize the polynomial fit
% here you can see how baseline noise really messes it up
figure, plot3(X(use, 1), X(use,2), ly, '.'); hold on
plot3(X(use, 1), X(use,2), Xd(use,:)*phat, '.')