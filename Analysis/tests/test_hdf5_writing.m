
%% test hd5 writing and reading

% specify parameters of dataset
NT = 1e3;
h = 20;
w = 25;
fname = 'test.hdf5';
if exist(fname, 'file')
    delete(fname)
end

% name of path to file system in the file
stim = 'Gabor';
grptrain = ['/' stim '/Train'];
grptest = ['/' stim '/Test'];

% create file
fprintf('Creating file [%s]\n', fname)

h5create(fname, [grptrain '/Stim'], [NT, h, w])
h5create(fname, [grptest '/Stim'], [NT, h, w])

SU = 1:10;
attr = 'SU';
h5writeatt(fname, ['/' stim], attr, SU)

disp('Writing data online in loop')
% loop over frames and write them online
xfull = rand([NT,h,w]);
for i = 1:NT
    x = xfull(i,:,:);
    h5write(fname, [grptrain '/Stim'], x, [i, 1, 1], [1 h w])
end

% read and check they all match

% Test 1: read all data
d = h5read(fname, [grptrain '/Stim']);
if all(xfull(:)==d(:))
    disp('Test 1 Passed. All data points match!')
end

% Test 2: read specified start point
start = 100;
count = [1, h, w];
d = h5read(fname, [grptrain '/Stim'], [start, 1, 1], count);
if all(all(xfull(start,:,:) == d))
    disp('Test 2 Passed. Loading specific datapoint matched!')
end

% Test 3: read attributes
d = h5readatt(fname, ['/' stim], attr);
if all(SU(:)==d(:))
    disp('Test 3 Passed. Loading attribute matched!')
end

fprintf('Cleaning up. Deleting [%s]\n', fname)
% delete(fname)

%% test writing every other line
% create file
fprintf('Creating file [%s]\n', fname)

h5create(fname, [grptrain '/Stim'], [NT, h, w])
h5create(fname, [grptest '/Stim'], [NT, h, w])

SU = 1:10;
attr = 'SU';
h5writeatt(fname, ['/' stim], attr, SU)

disp('Writing data online in loop')
% loop over frames and write them online
xfull = rand([NT,h,w]);
frameIdx = 1:10:NT;
for i = frameIdx
    x = xfull(i,:,:);
    h5write(fname, [grptrain '/Stim'], x, [i, 1, 1], [1 h w])
end

% read and check they all match

% Test 1: read all data
d = h5read(fname, [grptrain '/Stim']);

if all(all(all(xfull(frameIdx,:,:)==d(frameIdx,:,:))))
    disp('Test 4 Passed. All data points match using indexed entries!')
end

fprintf('Cleaning up. Deleting [%s]\n', fname)
delete(fname)