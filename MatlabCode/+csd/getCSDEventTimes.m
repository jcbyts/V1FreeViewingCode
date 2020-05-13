function eventTimes = getCSDEventTimes(Exp)
% gets the event times of the current source density trials
% Inputs:
%   Exp              [struct] - Exp struct from io.dataFactoryGratingSubspace
% 
% jfm wrote it 2020
% ghs edit it 2020

sanitycheck = 0; % check spike histogram to verify CSD activity

Trials = length(Exp.D);
CSD = [];
CSD.Trials = [];
for k = 1:Trials
   ProtoName = Exp.D{k}.PR.name;
   %ProtoName
   type = 0;
   if (strcmp(ProtoName,'ForageProceduralNoise'))  % for Jake
       type = 1;
   end
   if (strcmp(ProtoName,'Forage'))   % for Shanna
       type = 2;
   end
  % type
   if (type > 0)
       NoiseType = Exp.D{k}.PR.noisetype;
       %disp(sprintf('Trial(%d); %s  NoiseType(%d)',k,ProtoName,NoiseType));
       if (type == 1) && (NoiseType == 3)
           CSD.Trials = [CSD.Trials ; [k 1]];  % 1 indicate contrast onset
       end
       if (type == 2) && ( (NoiseType == 3) || (NoiseType == 6) ) 
           if (NoiseType == 3)
              CSD.Trials = [CSD.Trials ; [k 2]];
           end
           if (NoiseType == 6)
              CSD.Trials = [CSD.Trials ; [k 3]];
           end
       end
   end
end

CSD.Onsets = [];
CSD.Types = [];
CSD.MoDir = [];
CSD.Offsets = [];
CSD.Onsets_Ephys = [];
CSD.Offsets_Ephys = [];
NTrials = length(CSD.Trials);
for k = 1:NTrials
    kk = CSD.Trials(k,1);
    type = CSD.Trials(k,2);
    NoHist = Exp.D{kk}.PR.NoiseHistory;
    %*** Noise History is time in column 1, and contrast (0 or 1) in col 2
    %** find all the Onsets, as transition from 0 to 1 in column 2
    for i = 2:size(NoHist,1)
       if (NoHist(i-1,2) == 0) && (NoHist(i,2) >= 1)   % 
           tt = NoHist(i,1);
           CSD.Onsets = [CSD.Onsets ; tt];  % store Mat time
           CSD.Types = [CSD.Types ; type];  % 1 - contrast (Jake), 2 - contrast (Shanna), 3 - motion (Shanna)
           CSD.MoDir = [CSD.MoDir ; NoHist(i,2)];
           %******* convert to Ephys time per trial start-clocks
           if isfield(Exp.D{kk},'STARTCLOCKTIME')
               tt = tt - Exp.D{kk}.STARTCLOCKTIME;  % Jake - 0 from start of trial
           else
               tt = tt - Exp.D{kk}.eyeData(6,1);    % Shanna - start of trial in mat time
           end
           tt = tt + Exp.D{kk}.START_EPHYS;     % time from start of trial in ephys
           CSD.Onsets_Ephys = [CSD.Onsets_Ephys ; tt];
           % search for corresponding offset, if trial ends, insert a NaN
           for j = (i+1):size(NoHist,1)
             testOff = 0;
             if (type == 3)
                 testOff = (NoHist(j-1,2) >= 0) && (NoHist(j,2) < 0);
             else
                 testOff = (NoHist(j-1,2) >= 1) && (NoHist(j,2) == 0);
             end
             if (testOff)
               tt = NoHist(j,1);
               CSD.Offsets = [CSD.Offsets ; tt];  % store Mat time
               %******* convert to Ephys time per trial start-clocks
               if isfield(Exp.D{kk},'STARTCLOCKTIME')
                 tt = tt - Exp.D{kk}.STARTCLOCKTIME;  % 0 from start of trial
               else
                 tt = tt - Exp.D{kk}.eyeData(6,1);    % Shanna - start of trial in mat time 
               end
               tt = tt + Exp.D{kk}.START_EPHYS;     % time from start of trial in ephys
               CSD.Offsets_Ephys = [CSD.Offsets_Ephys ; tt];
               break;
             end
           end
           if (j >= size(NoHist,1))  % Offset occured after trial end
               CSD.Offsets = [CSD.Offsets ; NaN];
               CSD.Offsets_Ephys = [CSD.Offsets_Ephys ; NaN];
           end
       end 
    end
    %*** sanity check result looks right per trial
    if (sanitycheck == 1)
        figure(10); hold off;
        plot(NoHist(:,1),NoHist(:,2),'k-'); hold on;
        % plot(CSD.Onsets,(0.5*ones(size(CSD.Onsets))),'rx');
        xlabel('Time (secs)');
        %input('check');
    end
end


% Output (Other stuff was computed but we are ignoring that for now)
eventTimes = CSD.Onsets_Ephys;


if (sanitycheck == 1)
 for Unit = 1:size(Exp.sp,2) 
   NOnsets = length(CSD.Onsets_Ephys);
   SpkRaster = [];
   OffRaster = [];
   SpChan = Unit;
   for k = 1:NOnsets
     tt = CSD.Onsets_Ephys(k);
     z = find( (Exp.sp{SpChan}.st >= tt) & (Exp.sp{SpChan}.st < (tt+0.5) ) );
     if ~isempty(z)
      sptt = Exp.sp{SpChan}.st(z) - tt;  % time lock spikes relative to onset
      SpkRaster = [SpkRaster ; [(k*ones(size(sptt))) sptt]];
     end
     ott = CSD.Offsets_Ephys(k) - tt;
     OffRaster = [OffRaster ; [k ott]];
   end
   if ~isempty( SpkRaster )
     figure(10); hold off;
     plot(1000*SpkRaster(:,2),SpkRaster(:,1),'k.'); hold on;
     plot(1000*OffRaster(:,2),OffRaster(:,1),'r.'); hold on;
     xlabel('Time (ms)');
     ylabel('Trials');
     title(sprintf('CSD Raster: Unit(%d)',SpChan));
     input('check');
   else
     figure(10); hold off;
     
     disp(sprintf('Not plotting for unit %d, no spikes',Unit));
    end
 end
end
end

