%% align clocks
strobe_id = find(strcmp('DIGITAL-IN-07', {dig.board_dig_in_channels(:).native_channel_name}));
% get bit flip times
bittime = find(diff(dig.board_dig_in_data(strobe_id,:))>0)+1;
nbitflips = numel(bittime);
decval = zeros(nbitflips, 1);

%%
for i = 1:nbitflips
    ind = bittime(i);
    strobe_ = dig.board_dig_in_data(:,ind)';
    decval(i) = bin2dec(num2str( fliplr(strobe_(1:6)) ));
end

disp("Done")

% convert to sixlets
sixletoe0 = conv2(decval, fliplr(eye(6)));
sixletoe0 = sixletoe0(6:end,:);

strobeind = find(sixletoe0(:,end)==62) - 1;

strobetimes = bittime(strobeind)/30e3;
sixletoe0 = sixletoe0(strobeind,:);

remove = find(sum(diff(sixletoe0)==0, 2)==6)+1;

strobetimes(remove) = [];
sixletoe0(remove,:) = [];

sixletsptb = cell2mat(cellfun(@(x) x.STARTCLOCK, Exp.D, 'uni', 0));

% loop over trials and find matches
numTrials = numel(Exp.D);
matches = nan(numTrials,2);

oetime = nan(numTrials,2);
ptbtime = nan(numTrials,2);

for iTrial = 1:numTrials

    ptbtime(iTrial,1) = Exp.D{iTrial}.STARTCLOCKTIME;
    ptbtime(iTrial,2) = Exp.D{iTrial}.ENDCLOCKTIME;

    match0 = find(all(Exp.D{iTrial}.STARTCLOCK-sixletoe0 == 0 | Exp.D{iTrial}.STARTCLOCK-sixletoe0 == 1,2));
    match1 = find(all(Exp.D{iTrial}.ENDCLOCK-sixletoe0 == 0 | Exp.D{iTrial}.ENDCLOCK-sixletoe0 == 1,2));

    if numel(match0)>1
        [~, id] = min(abs(match0-matches(iTrial-1,1)));

        match0 = match0(id);
    end

    if numel(match1) > 1
        [~, id] = min(abs(match1-match0));
        match1 = match1(id);
    end
    
    if ~isempty(match0)
        matches(iTrial,1) = match0;
        oetime(iTrial,1) = strobetimes(match0);
    end
    
    if ~isempty(match1)
        matches(iTrial,2) = match1;
        oetime(iTrial,2) = strobetimes(match1);
    end    

    Exp.D{iTrial}.START_EPHYS = oetime(iTrial,1);
    Exp.D{iTrial}.END_EPHYS = oetime(iTrial,2);
    
end


oetime = reshape(oetime', [], 1);
ptbtime = reshape(ptbtime', [], 1);

% align clocks
Exp.ptb2Ephys = synchtime.align_clocks(ptbtime, oetime);

figure(1); clf
plot(ptbtime,oetime, 'o'); hold on
plot(ptbtime, Exp.ptb2Ephys(ptbtime))