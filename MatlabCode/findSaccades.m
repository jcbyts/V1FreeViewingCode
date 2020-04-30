function saccades = findSaccades(timestamps, xpos, ypos,varargin)
% Find saccades in eye position signals...
%
% Available arguments include:
%
%   order     - low-pass digital differentiating filter order (default: 32)
%   Wn        - low-pass filter corner freq, and transition band as a
%               percentage of the Nyquist frequency (default: [0.1,0.16])
%   accthresh - acceleration threshold (default: 2e4 deg./s^2)
%   velthresh - velocity threshold (default: 10 deg./s)
%   velpeak   - minimum peak velocity (default: 10 deg./s)
%   isi       - minimum inter-saccade interval (default: 0.050s)
%   debug     - show debugging output (default: false)
%
% See also: lpfirdd.

% 2016-11-13 - Shaun L. Cloherty <s.cloherty@ieee.org>

args = varargin;
p = inputParser;

p.addParameter('order',32,@(x) validateattributes(x,{'numeric'},{'scalar','even'})); % low-pass filter/differentiator order
p.addParameter('Wn',[0.1,0.16],@(x) validateattributes(x,{'numeric'},{'vector'})); % filter transition band (percentage of Nyquist frequency)

p.addParameter('accthresh',5e3,@(x) validateattributes(x,{'numeric'},{'scalar','positive'})); % accel. threshold for detection (deg./s^2) (was 2e4)
p.addParameter('velthresh',10,@(x) validateattributes(x,{'numeric'},{'scalar','positive'})); % velocity threshold (deg./s)
p.addParameter('velpeak',10,@(x) validateattributes(x,{'numeric'},{'scalar','positive'})); % min. peak velocity (deg./s)

p.addParameter('isi',0.050,@(x) validateattributes(x,{'numeric'},{'scalar','positive'}));
p.addParameter('dt',0.075,@(x) validateattributes(x,{'numeric'},{'scalar','positive'}));

p.addParameter('debug',false,@(x) validateattributes(x,{'logical'},{'scalar'}));

p.parse(args{:});

args = p.Results;

% low-pass FIR digital differentiator coefficients
N = round(args.order/2);
copt = lpfirdd(N, args.Wn(1), args.Wn(2), 1 ,0)';
coeffs = [fliplr(copt), 0, -copt];

timestamps = timestamps(:);
xpos = xpos(:);
ypos = ypos(:);

% get gaze position data...
t = timestamps;
pos = struct('x',xpos,'y',ypos);

fs = round(1/nanmedian(diff(timestamps))); % sampling freq. (samples/s)

% horiz. (x) and vert. (y) velocities
vel = structfun(@(x) fs*locfilt(coeffs,1,x),pos,'UniformOutput',false);

% scalar eye speed
speed = hypot(vel.x,vel.y);

% estimate baseline (e.g., pursuit) speed using a moving average...
a = 1;
b = ones(1,2*N+1)./(2*N+1);
baseline = locfilt(b,a,speed);
% baseline = circshift(baseline,N); % delay by half filter length

% scalar eye acceleration
accel = fs*locfilt(coeffs,1,speed);

% find saccades... i.e., -ve going zero crossings in the scalar eye
% acceleration signal. Note, quantizing the acceleration trace
% (i.e., dividing by 1e4) improves noise immunity
idx = findZeroCrossings(fix(accel)./args.accthresh,-1);

% ignore saccades with peak speed less than args.velpeak
idx(speed(idx) < args.velpeak) = [];

% ignore saccades too close to the start or end of the recording
idx(idx < args.order+args.dt*fs) = [];
idx(idx > length(t)-args.dt*fs) = [];

% sanity check...
idx(speed(idx) < baseline(idx)+args.velthresh) = [];

if isempty(idx)
    saccades=[];
    return
end

% % sanity check...
% k = bsxfun(@minus,idx,[1:args.dt*fs]-1);
% idx(~any(sign(fix(accel(k')./args.accthresh)) > 0.1)) = [];

n = ceil(args.isi*fs); % samples

% test for minimum inter-saccade interval (args.isi) violations?
if numel(idx) > 1
    ii = find(diff(idx) < n,1);
    while ~isempty(ii)
        % keep larger/faster of the two...
        [~,jj] = min(speed(idx(ii+[0,1])));
        idx(ii+jj-1) = [];
        ii = find(diff(idx) < n,1);
    end
end

if args.debug
    figure;
    subplot(2,1,1);
    plot(t,speed);
    hold on;
    
    subplot(2,1,2);
    plot(t,accel./args.accthresh);
    hold on;
    
    plot(t,fix(accel./args.accthresh));
end

if isempty(idx)
    tstart = [];
    tend = [];
    return
end

n = ceil(args.dt*fs); % samples

idx = kron(idx,ones(1,3)); % saccade indicies (start, mid, end)
for ii = 1:size(idx,1)
    t0 = max(idx(ii,2)-n,1);
    t1 = min(idx(ii,2)+n,length(t));
    
    % window speed signal (and apply threshold)
    tmp = speed(t0:t1) - args.velthresh;
    
    if args.debug
        clear fh
        for jj = 1:2
            subplot(2,1,jj);
            
            xx = t([t0,t1,t1,t0]);
            yy = kron(get(gca,'YLim'),[1,1]);
            fh(jj) = fill(xx(:),yy(:),zeros(1,3));
            set(fh,'ZData',-1*ones(size(xx)),'FaceAlpha',0.1,'LineStyle','none');
        end
        
        subplot(2,1,1);
        plot(t([t0,t1]),args.velthresh*[1,1],'k--');
        plot(t([t0:t1]),baseline(t0:t1)+args.velthresh,'k-');
    end
    
    % now search back in time to find saccade start, accounting
    % for any baseline/pursuit speed
    k = findZeroCrossings(tmp-baseline(t0:t1),1);
    if isempty(k)
        continue
    end
    
    k(k > n+1) = [];
    %   if ~isempty(k)
    idx(ii,1) = max(k) + t0 - 1; % index of saccade start
    %   end
    
    % search forward in time to find saccade end
    k = findZeroCrossings(tmp-baseline(idx(ii,1)),-1);
    k(k < n+1) = [];
    if ~isempty(k)
        idx(ii,3) = min(k) + t0 - 1; % index of saccade end
    end
    
    if args.debug
        xx = t(idx(ii,[1,3,3,1]));
        arrayfun(@(h) set(h,'XData',xx),fh);
    end
    
    % the code below attempts to refine our estimate of saccade start
    % and end by computing eye velocity in the direction of the saccade,
    % i.e., by projecting x and y velocity components onto the saccade
    % vector.
    %
    % if this fails, we should go with our current estimate...
    
    % determine saccade direction vector...
    v = [diff(pos.x(idx(ii,[1,3]))); diff(pos.y(idx(ii,[1,3])))]; % FIXME: should dy be -ve?
    v = v./norm(v); % unit vector
    
    % project velocity onto the saccade vector
    t0 = max(idx(ii,1)-(2*N+1),1);
    t1 = min(idx(ii,3)+(2*N+1),length(t));
    
    sacvel = [vel.x(t0:t1), vel.y(t0:t1)]*v;
    
    % apply threshold
    tmp = sacvel - args.velthresh;
    
    % estimate baseline/pursuit speed at saccade start
    sacvel_ = nanmean(sacvel(1:2*N+1));
    
    % refine estimate of saccade start and end
    k = findZeroCrossings(tmp-sacvel_,1);
    k(k+t0-1 > idx(ii,2)) = [];
    if ~isempty(k)
        idx(ii,1) = max(k) + t0 - 1; % index of saccade start
    else
        % go with our current estimate...
        if args.debug
            warning('Failed to refine estimate of start time for saccade at t = %3f.',timestamps(idx(ii,2)));
        end
    end
    
    % estimate baseline/pursuit speed at saccade end
    sacvel_ = nanmean(sacvel(end-(2*N+1):end));
    
    k = findZeroCrossings(tmp-sacvel_,-1);
    k(k+t0-1 < idx(ii,2)) = [];
    if ~isempty(k)
        idx(ii,3) = min(k) + t0 - 1; % index of saccade end
    else
        % go with our current estimate...
        if args.debug
            warning('Failed to refine estimate of end time for saccade at t = %3f.',timestamps(idx(ii,2)));
        end
    end
    
    if args.debug
        xx = t(idx(ii,[1,3,3,1]));
        arrayfun(@(h) set(h,'XData',xx),fh);
        drawnow
    end
    
end

tstart = t(idx(:,1));
tend = t(idx(:,3));
idx(:,2) = [];

saccades = struct();
saccades.tstart = tstart(:);
saccades.tend   = tend(:);
saccades.duration = saccades.tend- saccades.tstart;
saccades.startIndex = idx(:,1);
saccades.endIndex   = idx(:,end);
saccades.startXpos  = xpos(saccades.startIndex);
saccades.startYpos  = ypos(saccades.startIndex);
saccades.endXpos  = xpos(saccades.endIndex);
saccades.endYpos  = ypos(saccades.endIndex);
saccades.dX     = saccades.endXpos - saccades.startXpos;
saccades.dY     = saccades.endYpos - saccades.startYpos;
saccades.size   = sqrt(saccades.dX.^2 + saccades.dY.^2);
n = numel(saccades.tstart);
saccades.vel    = zeros(n, 1);
saccades.peakIndex    = zeros(n, 1);
saccades.tpeak    = zeros(n, 1);
for i = 1:n
    [v,ind] = max(speed(saccades.startIndex(i):saccades.endIndex(i)) );
    saccades.vel(i) = v;
    saccades.peakIndex(i) = saccades.startIndex(i) + ind - 1;
    saccades.tpeak(i) = timestamps(saccades.peakIndex(i));
end


%   result (struct)
%       .tstart         saccade start time
%       .tend           saccade end time
%       .duration      saccade duration
%       .size          saccade size
%       .startXpos     x position at start
%       .startYpos     y position at start
%       .endXpos       x position at end
%       .endYpos       y position at end
%       .startIndex    start index (into data trace)
%       .endIndex      end index

% vel = speed;
acc = accel;

    function y = locfilt(b,a,x)
        y = filter(b,a,x);
        y(1:2*N) = NaN;
        y = circshift(y,-N);
    end



%---------------------------------------------------------
% Low-pass FIR digital differentiator (LPFIRDD) design   -
% via constrained quadratic programming (QP)             -
% [Copt,c]=lpfirdd(N,alpha,beta,r,idraw)                 -
% By Dr Yangquan Chen		019-07-1999                   -
% Email=<yqchen@ieee.org>; URL=http://www.crosswinds.net/~yqchen/
% --------------------------------------------------------
% LPFIRDD: only 1st order derivative estimate
% total taps=2N. c(i)=-c(i+N+1); c(N+1)=0 (central point)
%
%          -N      -N+1            -1                    N
% FIR=c(1)z   +c(2)z    +...+ c(N)z  + 0 + ... + c(2N+1)z
%
%       N
%     ------
%     \                    j   -j
%      >       Copt(j) * (z - z  )
%     /
%     ------
%      j=1
% N: Taps  (N=2, similar to sgfilter(2,2,1,1)
% alpha ~ beta: transit band of frequency
%				    (in percentage of Nyquest freq)
% r: the polynomial order. r<=N Normally, set it to 1.
%---------------------------------------------------------------
    function [Copt,bd]=lpfirdd(N,alpha,beta,r,idraw)
        % testing parameters
        % alpha=1./pi;beta=1.5/pi;N=10;r=1;idraw=1;
        if (alpha>beta)
            disp('Error in alpha (alpha<=beta)');return;
        end
        if ((beta>1) || (beta <0))
            disp('Error in Beta! (beta in [0,1]');return;
        end
        if ((alpha>1) || (alpha <0))
            disp('Error in Alpha! (Alpha in [0,1]');return;
        end
        % default r=1
        if (r<1); r=1; end
        
        % matrix W
        W=zeros(r,N);
        for ix=1:N
            for jx=1:r
                W(jx,ix)=ix^(2*jx-1);
            end
        end
        
        %matrix L
        L=zeros(N,1);
        if (beta>alpha)
            for ix=1:N
                L(ix)=(alpha*sin(ix*beta*pi)-beta*sin(ix*alpha*pi))/ix/ix/(beta-alpha);
            end
        elseif (beta==alpha)
            for ix=1:N
                L(ix)=(ix*alpha*pi*cos(ix*alpha*pi)-sin(ix*alpha*pi))/ix/ix;
            end
        end
        % matrix e
        ex=zeros(r,1);ex(1)=1;
        % optimal solution
        % Copt=W'*inv(W*W')*(ex + 2.*W*L/pi)-2.*L/pi;
        Copt=W'*pinv(W*W')*(ex + 2.*W*L/pi)-2.*L/pi;
        Copt=Copt/2;
        % fr plots
        if (idraw==1)
            bd=[-fliplr(Copt'),0,Copt']';
            %ad=1;sys_sg=tf(bd',ad,1./Fs);bode(sys_sg)
            Fs=12790;nL=N;nR=N;npts=1000;%w=logspace(0,4,npts);
            w=((1:npts)-1)*pi/npts;
            j=sqrt(-1);ejw=zeros(nL+nR+1,npts);
            for ix=(-nL:nR)
                ejw(ix+nL+1,:)=exp(j*ix*w);
            end
            freq=bd'*ejw;
            figure;subplot(2,1,1)
            plot(w/pi*Fs/2,(abs(freq)));grid on;
            hold on; ax=axis;ax(2)=Fs/2;axis(ax);
            xlabel('freq. (Hz)');ylabel('amplitude (dB)');
            subplot(2,1,2)
            plot(w/pi*Fs/2,180*(angle(freq))/pi );grid on;
            hold on; ax=axis;ax(2)=Fs/2;axis(ax);
            xlabel('freq. (Hz)');ylabel('phase anlge (deg.)');
            
            figure;subplot(2,1,1);Fs=12600; % Hz for U8
            semilogx(w*Fs/pi/2,20*log10(abs(freq)));grid on;
            hold on; ax=axis;ax(2)=Fs/2;axis(ax);
            semilogx([Fs/2,Fs/2],[ax(3),ax(4)],'o-r');grid on;
            xlabel('freq. (Hz)');ylabel('amplitude (dB)');
            subplot(2,1,2)
            semilogx(w*Fs/pi/2,180*(angle(freq))/pi );grid on;
            hold on; ax=axis;ax(2)=Fs/2;axis(ax);
            semilogx([Fs/2,Fs/2],[ax(3),ax(4)],'o-r');grid on;
            xlabel('freq. (Hz)');ylabel('phase anlge (deg.)');
        end
    end

    function indices = findZeroCrossings(data, mode)
        %FINDZEROCROSSINGS Find zero crossing points.
        %   I = FINDZEROCROSSINGS(DATA,MODE) returns the indicies into the supplied
        %   DATA vector, corresponding to the zero crossings.
        %
        %   MODE specifies the type of crossing required:
        %     MODE < 0 - results in indicies for the -ve going zero crossings,
        %     MODE = 0 - results in indicies for ALL zero crossings (default), and
        %     MODE > 0 - results in indicies for the +ve going zero crossings.
        
        % $Id: findZeroCrossings.m,v 1.1 2008-07-21 23:31:50 shaunc Exp $
        
        if nargin < 2
            mode = 0;
        end
        
        [indices,~,p0] = find(data); % ignore zeros in the data vector
        
        switch sign(mode)
            case -1
                % find -ve going crossings
                iit = find(diff(sign(p0))==-2);
            case 0
                % find all zero crossings
                iit = find(abs(diff(sign(p0)))==2);
            case 1
                % find +ve going crossings
                iit = find(diff(sign(p0))==2);
        end
        
        indices = round((indices(iit)+indices(iit+1))/2);
    end

end




