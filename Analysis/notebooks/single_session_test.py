
# %% list sessions
sesslist = gt.list_sessions()
sesslist = list(sesslist)
for i in range(len(sesslist)):
    print("%d %s" %(i, sesslist[i]))
#%%
sess = [sesslist[33]]
stim, _, _, Robs, _, basis, opts, sacbc, valid, eyepos = gt.load_and_setup(sess,npow=1.8)
matdat = gt.load_data(sess[0])
ft = matdat['grating']['frameTime']
len(ft)

#%%



# get valid saccade times
sacon = np.where(np.diff(sacbc, axis=0)==1)[0]
validsaccades = np.intersect1d(sacon, np.where(valid)[0])
print("%d/%d valid samples" %(len(validsaccades), len(sacon)))

# run_sac_triggered_analyses(sess)

# %%
sonset = matdat['slist'][:,0]
sonsetAll = sonset.copy()
s0 = np.min(validtimes)
s1 = np.max(validtimes)
sonset = sonset[sonset > s0]
sonset = sonset[sonset < s1
vinds = np.where(np.sum(np.abs(np.expand_dims(sonset, axis=1) - validtimes) < 1/opts['frate'], axis=1))[0]
sonset = sonset[vinds
win = [-300, 300
# bin spike times at 1ms resolution
spbins = np.arange(s0-win[0]/1e3, s1+win[1]/1e3, 1e-3
# index into saccade by statistics
ind = np.digitize(sonset+1e-3, sonsetAll)-1
off = matdat['slist'][ind,4].astype(int)
on = matdat['slist'][ind,3].astype(int)
dx = matdat['eyepos'][off,1] - matdat['eyepos'][on,1]
dy = matdat['eyepos'][off,2] - matdat['eyepos'][on,2]
sacAmp = np.hypot(dx, dy)
soffset = matdat['slist'][ind,1]