
## Free Viewing
Code to analyze marmoset V1 data from the Mitchell lab.

### Getting started
First thing necessary is to setup your user paths in `addFreeViewingPaths`

```
edit addFreeViewingPaths
```

This function has a big switch statement that checks which user is calling and sets the paths appropriately. You need to create a new case for your user. You need the stimulus code (marmoView), which can be found [here](https://github.com/jcbyts/MarmoV5). If you don't have permission, email Jake to ask for it.

Set the path to marmoview and the path to the data. You don't need marmopipe or `PREPROCESSED_DATA_DIR` unless you're importing raw data.

``` matlab
case 'jakelaptop'
        % we need marmoview in the path for the stimulus objects to
        % regenerate properly
        marmoViewPath = '~/Documents/MATLAB/MarmoV5/';
        
        % you only need marmopipe if importing raw data
        marmoPipePath = [];

        % where the data live
        dataPath = '~/Dropbox/Projects/FreeViewing/Data';

        % processed data:
        setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', dataPath)
```

Then call:
```
user = 'yourusername'
addFreeViewingPaths(user);
```

### Regenerating stimuli for analysis
Use the function `io.dataFactory` to load sessions. Call it without any arguments to see all available sessions.
Call it with a number id input or a string id and you'll get back an `Exp` struct and an options struct.

``` matlab
    sessId = 'L20191205'
    [Exp, S] = io.dataFactory(sessId);
```


Then use the function `dataGenerate` to regenerate the stimulus. It will save a file out with the required matrices to run NIM or NDN.

#####Example 1:
This will regenerate the stimulus for the Gabor stimulus set using the default parameters
``` matlab
    fname = io.dataGenerate(Exp, S, 'stimulus', 'Gabor');
```


#####There are multiple optional arguments:
    
    ('stimulus', 'Gabor')       % BackImage, Dots, Gabor,Grating
    ('testmode', true)          % true or false (uses 10 trials if true)
    ('fft', false)              % converts stimulus to fourier energy
    ('eyesmooth', 3)            % smoothing on eye position (using sgolay filter, must be an odd number)
    ('t_downsample', 1)         % temporal downsampling
    ('s_downsample', 1)         % spatial downsampling
    ('includeProbe', true)      % true or false, include the probe stimuli in the reconstruction (slower, but more accurate, still not completely debugged)

#####Example 2:
Recreate the Gabor stimulus with downsampling and smoothing on the eye position.

``` matlab
fname = io.dataGenerate(Exp, S, ...
'stimulus', 'Gabor', ...
't_downsample', 2, ...
's_downsample', 2, ...
'eyesmooth', 5)

load(fname); % load all the stimuli

% plot the sacccade-triggered average firing rate for all units
[an, ~, widx] = eventTriggeredAverage(mean(Robs,2), slist(:,2), [-1 1]*ceil(.2/dt));
figure(1); clf
plot(widx * dt, an / dt)

% spike triggered average stimulus for 25 units
figure(2); clf
ix = valdata == 1 & labels == 1 & probeDist > 50;
nlags = ceil(.1/dt);
for k = 1:min(25, size(Robs,2))
    sta = simpleSTC((stim(ix,:)), Robs(ix,k), nlags );

    subplot(5, 5, k, 'align')
    imagesc(sta)
end
```