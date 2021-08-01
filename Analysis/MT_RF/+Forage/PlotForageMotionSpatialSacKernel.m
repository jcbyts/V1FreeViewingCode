function PlotForageMotionSpatialSacKernel(rfinfo,ftag)
% function PlotForageSpatialKernel(rfinfo,ftag)
%   shows the results of RF analysis as plots, show file tag for unit name

%******* take all fields of rfinfo and make part of the environment
%******** download variables stored in info
fields = fieldnames(rfinfo);
for k = 1:size(fields,1)
      str = [fields{k} ' = rfinfo.' fields{k} ';'];
      eval(str);
end
%*******************************
  
%********* Plotting routines below ***************
if (1)
  hf = figure;
  set(hf,'position',[150 50 900 900]);
  subplot('position',[0.1 0.75 0.25 0.20]);
  plot(tSXX,mvec(SACN+1,:),'k-'); hold on;
  plot(tSXX,mvec(SACN+1,:) + (2*mvec2(SACN+1,:)),'k-'); hold on;
  plot(tSXX,mvec(SACN+1,:) - (2*mvec2(SACN+1,:)),'k-'); hold on;
  axis tight;
  V = axis;
  plot([0,0],[V(3),V(4)],'k-');
  plot([POSTPERIOD(1),POSTPERIOD(1)],[V(3),V(4)],'b-');
  plot([POSTPERIOD(2),POSTPERIOD(2)],[V(3),V(4)],'b-');
  plot([DURPERIOD(1),DURPERIOD(1)],[V(3),V(4)],'m-');
  plot([DURPERIOD(2),DURPERIOD(2)],[V(3),V(4)],'m-');
  plot([PREPERIOD(1),PREPERIOD(1)],[V(3),V(4)],'r-');
  plot([PREPERIOD(2),PREPERIOD(2)],[V(3),V(4)],'r-');
  plot([tSXX(1),tSXX(end)],[sacbase,sacbase],'k-');
  plot([tSXX(1),tSXX(end)],[sacbase+(2*sacbasestd),sacbase+(2*sacbasestd)],'k-');
  plot([tSXX(1),tSXX(end)],[sacbase-(2*sacbasestd),sacbase-(2*sacbasestd)],'k-'); 
  xlabel('Time (ms)');
  ylabel('Firing Rate');
  title(sprintf('Avg Sac Mod, %s',ftag));
  subplot('position',[0.4 0.75 0.25 0.20]);
  for k = 1:SACN
    plot(tSXX,mvec(k,:),'k-'); hold on;
  end
  axis tight;
  V = axis;
  plot([0,0],[V(3),V(4)],'k-');
  plot([POSTPERIOD(1),POSTPERIOD(1)],[V(3),V(4)],'b-');
  plot([POSTPERIOD(2),POSTPERIOD(2)],[V(3),V(4)],'b-');
  plot([DURPERIOD(1),DURPERIOD(1)],[V(3),V(4)],'m-');
  plot([DURPERIOD(2),DURPERIOD(2)],[V(3),V(4)],'m-');
  plot([PREPERIOD(1),PREPERIOD(1)],[V(3),V(4)],'r-');
  plot([PREPERIOD(2),PREPERIOD(2)],[V(3),V(4)],'r-');
  xlabel('Time (ms)');
  ylabel('Firing Rate');
  title('Average Saccade Modulation');
  subplot('position',[0.7 0.75 0.25 0.20]);
  xx = (0:(ND+6))*(360/ND);
  yy = [1:ND,1:(size(xx,2)-ND)];
  plot(xx,postcurve(yy),'b-'); hold on;
  plot(xx,postcurve(yy)+(2*postcurve_std(yy)),'b-'); hold on;
  plot(xx,postcurve(yy)-(2*postcurve_std(yy)),'b-'); hold on;
  plot(xx,durcurve(yy),'m-'); hold on;
  plot(xx,durcurve(yy)+(2*durcurve_std(yy)),'m-'); hold on;
  plot(xx,durcurve(yy)-(2*durcurve_std(yy)),'m-'); hold on;
  plot(xx,precurve(yy),'r-'); hold on;
  plot(xx,precurve(yy)+(2*precurve_std(yy)),'r-'); hold on;
  plot(xx,precurve(yy)-(2*precurve_std(yy)),'r-'); hold on;
  axis tight;
  plot([xx(1),xx(end)],[sacbase,sacbase],'k-');
  plot([xx(1),xx(end)],[sacbase+(2*sacbasestd),sacbase+(2*sacbasestd)],'k-');
  plot([xx(1),xx(end)],[sacbase-(2*sacbasestd),sacbase-(2*sacbasestd)],'k-'); 
  xlabel('Direction (degs)');
  ylabel('Saccade Rate');
  title('Saccade Mod by Direction');
end

%**** plot RF across time frames, and show significant points
if (1)
    dx = (0.9/KN);
    dxs = (0.9/(KN+1));
    mino = min(min(mcounts));
    maxo = max(max(mcounts));
    for it = 1:(KN-1)
       fx = 0.075 + ((it-1)*dx);
       subplot('position',[fx 0.55 dxs dxs]);
       ito = (2-DTA)+it;
       svec = flipud( reshape(squeeze(mcounts(:,ito)),Nx,Ny)' );
       imagesc(Zx,Zy,svec,[mino maxo]); hold on;
       plot([-11,11],[0,0],'k-');
       plot([0,0],[-11,11],'k-');
       axis off;
       h = title(sprintf('%4.1f',-tXX(ito)));
       %********* mark sig locations if any
       subplot('position',[fx 0.45 dxs dxs]);
       svec = flipud( reshape(squeeze(sigcounts(:,ito)),Nx,Ny)' );
       imagesc(Zx,Zy,svec,[-1 1]); hold on;
       plot([-11,11],[0,0],'r-');
       plot([0,0],[-11,11],'r-');
       axis off;
       %***********
    end
    fx = 0.075 + ((KN-1)*dx);
    subplot('position',[fx 0.55 dxs dxs]);
    imagesc(Zx,Zy,ones(size(svec))*mino,[mino maxo]); hold on;
    axis off;
    h = colorbar;
    
    %***********
    subplot('position',[0.075 0.15 0.20 0.20]);
    imagesc(Zx,Zy,meanrf,[mino maxo]); hold on;
    plot([-11,11],[0,0],'k-');
    plot([0,0],[-11,11],'k-');
    axis off;
    title('Mean RF');
    %************   
   
    %****** plot RFs by sparseness
    if (NS > 1)
        dx = (0.4/(NS/2));
        dxs = (0.4/((NS/2)+1));
        for sp = 1:NS
          ix = mod((sp-1),2);
          iy = 1-floor((sp-1)/2);
          fx = 0.30 + (ix*dx);
          fy = 0.05 + (iy*dx);
          subplot('position',[fx fy dxs dxs]);
          imagesc(Zx,Zy,sparserf{sp},[mino maxo]); hold on;
          plot([-11,11],[0,0],'k-');
          plot([0,0],[-11,11],'k-');
          axis off;
          title(sprintf('Sparse=%d',spark(sp)));
        end
    end
    
    %********** plot motion selectivity over sig points
    subplot('position',[0.7 0.075 0.25 0.3]);
    xx = (0:(ND+6))*(360/ND);
    yy = [1:ND,1:(size(xx,2)-ND)];
    plot(xx,mou(yy),'k-'); hold on;
    plot(xx,mou(yy)+(2*mostd(yy)),'k-'); hold on;
    plot(xx,mou(yy)-(2*mostd(yy)),'k-'); hold on;
    plot([xx(1),xx(end)],[sacbase,sacbase],'k-');
    plot([xx(1),xx(end)],[sacbase+(2*sacbasestd),sacbase+(2*sacbasestd)],'k-');
    plot([xx(1),xx(end)],[sacbase-(2*sacbasestd),sacbase-(2*sacbasestd)],'k-'); 
    %***** replot single lines from sac modulation for comparison
    plot(xx,postcurve(yy),'b-'); hold on;
    plot(xx,durcurve(yy),'m-'); hold on;
    plot(xx,precurve(yy),'r-'); hold on;
    %**********
    axis tight;
    xlabel('Direction (degs)');
    ylabel('Firing (hz)');
    title('Motion selectivity');
    %***************
end

return;
