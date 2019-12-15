
## Free Viewing
Code to analyze marmoset V1 data from the Mitchell lab.

### Getting started
First thing necessary is to setup your user paths in `addFreeViewingPaths`. If you're working in the terminal, use your favorite editor to edit `addFreeViewingPaths`.

```
vim addFreeViewingPaths
```

This function has a big switch statement that checks which user is calling and sets the paths appropriately. You need to create a new case for your user. You need the stimulus code (marmoView), which can be found [here](https://github.com/jcbyts/MarmoV5). If you don't have permission, email Jake to ask for it.

Set the path to marmoview and the path to the data. You don't need marmopipe or `SERVER_DATA_DIR` unless you're importing raw data.

``` matlab
case 'jakesigur'
    % we need marmoview in the path for the stimulus objects to
    % regenerate properly
    marmoViewPath = '~/Repos/MarmoV5/';
    % we only need marmopipe to import raw data
    marmoPipePath = [];
    % where the data live
    dataPath = '~/Data';
        
    % add data path as a preference
    setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', dataPath)
```

Then run matlab. Call:
```
user = 'yourusername'
addFreeViewingPaths(user);
```

Make sure you can load data.
``` matlab
    sessId = 'L20191205'
    [Exp, S] = io.dataFactory(sessId);
```
Make sure everything is running properly. Use the function `dataGenerate` to regenerate the stimulus for 5 trials.

``` matlab
    fname = io.dataGenerate(Exp, S, 'stimulus', 'Gabor', 'testmode', 5);
```

### Regenerating stimuli for analysis
To see this all in action, use the script `Analysis/201912_example/exampleScript.m` and step through the sections.

Otherwise, follow along here.

Use the function `io.dataFactory` to load sessions. Call it without any arguments to see all available sessions.
Call it with a number id input or a string id and you'll get back an `Exp` struct and an options struct.

``` matlab
    sessId = 'L20191205'
    [Exp, S] = io.dataFactory(sessId);
```


Use the function `dataGenerate` to regenerate the stimulus. It will save a file out with the required matrices to run NIM or NDN.

The ouput of the function is the name of the file with the saved stimulus. That file contains the following fields:
    
    NX      x-spatial dimension of the stimulus for reshaping
    stim    [Ntime x Ndims] the stimulus
    Robs    [Ntime x Nunits] binned spike counts        
    dt      [1 x 1] the temporal bin size (in seconds)    
    labels  [Ntime x 1] each sample eye state (1: fixation, 2: saccade, 3: blink, 4:lost track)         
    opts    [struct] the options that were used to call dataGenerate        
    probeDist [Ntime x 1] distance of the probe from the center of the ROI in stim (in pixels)
    slist   [Nsaccades x 2] saccade start and stop index
    valdata [Ntime x 1] valid times (seed,frame time)      
    xax     [1 x nx] values of the x-dimension        
    yax     [1 x ny] values of the y-dimension


``` matlab
    fname = io.dataGenerate(Exp, S, 'stimulus', 'Gabor', 'testmode', 5);
```


##### There are multiple optional arguments:
    
    ('stimulus', 'Gabor')       % BackImage, Dots, Gabor,Grating
    ('testmode', true)          % true or false (uses 10 trials if true)
    ('fft', false)              % converts stimulus to fourier energy
    ('eyesmooth', 3)            % smoothing on eye position (using sgolay filter, must be an odd number)
    ('t_downsample', 1)         % temporal downsampling
    ('s_downsample', 1)         % spatial downsampling
    ('includeProbe', true)      % true or false, include the probe stimuli in the reconstruction (slower, but more accurate, still not completely debugged)

##### Example:
Recreate the Gabor stimulus with downsampling and smoothing on the eye position.

``` matlab
fname = io.dataGenerate(Exp, S, ...
'stimulus', 'Gabor', ...
't_downsample', 2, ...
's_downsample', 2, ...
'eyesmooth', 5)
```

Load the output and run some simple analyses
``` matlab
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