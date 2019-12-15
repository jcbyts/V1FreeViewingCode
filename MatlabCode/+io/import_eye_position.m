function Exp = import_eye_position(Exp, DataFolder, varargin)
% Exp = import_eye_position(Exp, DataFolder, varargin)

ip = inputParser();
ip.addParameter('TAGSTART', 63)
ip.addParameter('TAGEND', 62)
ip.parse(varargin{:});

TAGSTART = ip.Results.TAGSTART;
TAGEND = ip.Results.TAGEND;


%% Loading up the VPX file as a long data stream
% Can be done later, might just use matlab eye data for now
%*****************************************************************
DDPI = 0;
EDF = 0;
VpxFiles = dir([DataFolder,filesep,'*.vpx']);
if isempty(VpxFiles)
    % try to find ddpi files if no VPX files
    VpxFiles = dir([DataFolder,filesep,'*.ddpi']);
    if isempty(VpxFiles)
        %******** if not DDPI, then try to find .edf from EyeLink
        VpxFiles = dir([DataFolder,filesep,'*edf.mat']);
        if isempty(VpxFiles)
            disp('Error finding raw eye data file');
            return
        else
            EDF = 1;
        end
    else
        DDPI = 1;
    end
end

%****** get order by date of files to import them
FileDates = cellfun(@datenum,{VpxFiles(:).date});
DoFileSort = [ (1:size(FileDates,2))' FileDates'];
FileSort = sortrows(DoFileSort,2); % sort them by date
%***** read and append files in order by date *********
BigN = 0;

for zk = FileSort(:,1)'
    fname = VpxFiles(zk).name;
    vpx_filename = [DataFolder,filesep,VpxFiles(zk).name];
    if (DDPI == 1)
        vpx = read_ddpi.load_ddpi_file(vpx_filename);  % makes DDPI look like VPX
    elseif (EDF==1)
        vpx = read_edf.load_edf_file(vpx_filename);
    else
        vpx = read_vpx.load_vpx_file(vpx_filename);
    end
    if ~isempty(vpx)
        if ~BigN
            Exp.vpx = vpx;
        else
            if ~(DDPI || EDF)
                %******* append time to prevent overlap across files
                vpx.raw(:,1) = vpx.raw(:,1) + BigN;  % add time offset before concat
                vpx.smo(:,1) = vpx.smo(:,1) + BigN;
                vpx.tstrobes = vpx.tstrobes + BigN;
            end
            %******* concatenate large file stream **********
            Exp.vpx.raw = [Exp.vpx.raw ; vpx.raw];
            Exp.vpx.smo = [Exp.vpx.smo ; vpx.smo];
            Exp.vpx.tstrobes = [Exp.vpx.tstrobes ; vpx.tstrobes];
            Exp.vpx.strobes = [Exp.vpx.strobes ; vpx.strobes];
            %******** compute new last time, plus one minute
        end

        BigN = Exp.vpx.smo(end,1) + 60.0; % last time point plus one minute
        
        
        clear vpx;
        disp('******************************************');
        fprintf('Experiment file %s loaded\n',fname);
    else
        fprintf('WARNING: failed to read %s\n',fname);
    end
end


%% Synching up the strobes from VPX to MarmoView
% Same thing, might use matlab eye data for now
%*****************************************************************
disp('Synching up vpx strobes');
for k = 1:size(Exp.D,1)
    start = synchtime.find_strobe_time(TAGSTART,Exp.D{k}.STARTCLOCK,Exp.vpx.strobes,Exp.vpx.tstrobes);
    finish = synchtime.find_strobe_time(TAGEND,Exp.D{k}.ENDCLOCK,Exp.vpx.strobes,Exp.vpx.tstrobes);
    if (isnan(start) || isnan(finish))
        fprintf('Synching VPX trial %d\n',k);
        if isnan(finish) && isnan(start)
            fprintf('Dropping entire VPX trial %d from protocol %s\n',k,Exp.D{k}.PR.name);
            Exp.D{k}.START_VPX = NaN;
            Exp.D{k}.END_VPX = NaN;
        else
            %******* here we could try to repair a missing code, or find a
            %******* a partial code nearby
            Exp.D{k}.START_VPX = start;
            Exp.D{k}.END_VPX = finish;
            tdiff = Exp.D{k}.eyeData(end,6) - Exp.D{k}.eyeData(1,1);
            if isnan(start) && ~isnan(finish)
                Exp.D{k}.START_VPX = Exp.D{k}.END_VPX - tdiff;
                disp('Approximating VPX start code');
            end
            if isnan(finish) && ~isnan(start)
                Exp.D{k}.END_VPX = Exp.D{k}.START_VPX + tdiff;
                disp('Approximating VPX end code');
            end
            %****************
        end
    else
        Exp.D{k}.START_VPX = start;
        Exp.D{k}.END_VPX = finish;
    end
end
disp('Finished synching up vpx strobes');
