#%% use loco workspace in conda

import os, sys
import h5py
import matplotlib
from numpy.lib.arraysetops import unique
sys.path.insert(0, '/mnt/Data/Repos/')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d
%matplotlib ipympl
# %matplotlib inline
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from neureye.models.display import animshow

#%% load huklab dataset
import data_factory as dfac
sessions = dfac.get_huklab_sessions()

num_sessions = len(sessions[0])


session = dfac.get_huklab_session(sessions[1], sessions[0], 0)

D = dfac.process_huklab_session(session)

#%% load allen dataset
import os
import data_factory as dfac
sessions, cache = dfac.get_allen_sessions()


downloaded_sessions = [f for f in os.listdir('/mnt/Data/Datasets/allen/ecephys_cache_dir/') if 'session_' in f]

#%%
i = 0
sessid = np.where(int(downloaded_sessions[i][8:]) == sessions.index.values)[0][0]
print(sessid)

#%%
todownload = np.where(sessions.session_type=='functional_connectivity')

downloaded_index = [np.where(int(downloaded_sessions[i][8:]) == sessions.index.values)[0][0] for i in range(len(downloaded_sessions))]
print(downloaded_index)
#%%
sessid = 27

#%%
session = dfac.get_allen_session(cache, sessions, sessid)
# D = dfac.process_allen_dataset(session)

#%%
%matplotlib ipympl
plt.figure()
onsets = D.gratings.start_time.to_numpy()
igrat = 1

ax1 = plt.subplot(3,1,1)
plt.plot(D.eye_data['eye_time'], D.eye_data['eye_y'], color='k')

nsac = len(D.saccades['start_time'])
for ii in range(nsac):
    iix = np.arange(D.saccades['start_index'][ii], D.saccades['stop_index'][ii], 1, dtype=int)
    plt.plot(D.eye_data['eye_time'][iix], D.eye_data['eye_y'][iix], color='r')

for i in range(len(onsets)):
    plt.axvline(onsets[i], color='r', linestyle='--')

ax1.set_xlim( (-1 + onsets[igrat], onsets[igrat] + 60))
ax1.set_ylim( (-10, 10))

ax2 = plt.subplot(3,1,2)
plt.plot(D.eye_data['eye_time'], D.eye_data['pupil'], color='k')
ax2.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))
# ax2.set_ylim((0, .005))

ax3 = plt.subplot(3,1,3)
plt.plot(D.run_data['run_time'], D.run_data['run_spd'], color='k')
for i in range(len(D.run_epochs['start_time'])):
    plt.axvspan(D.run_epochs['start_time'][i], D.run_epochs['stop_time'][i], color='gray', alpha=.5)
ax3.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))
ax3.set_ylim((0, 45))

#%%
igrat = igrat + 10
for ax in [ax1, ax2, ax3]:
    ax.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))
    ax3.set_ylim((0, 45))
#%% Make animation
%matplotlib inline
anim = dfac.make_animation(D)

print("Saving... this can be slow")
anim.save(D.files.session + '.mp4', fps=10)
print("Done")


#%% helper functions for main analysis
%matplotlib inline
def save_analyses(D, fpath='./analyses/'):
    import pickle
    fname = os.path.join(fpath, D.files.session + '.pkl')
    # save the data to a file with pickle
    with open(fname, 'wb') as f:
        pickle.dump(D, f)

def load_analyses(session_name, fpath='./analyses/'):
    import pickle
    fname = os.path.join(fpath, session_name + '.pkl')
    D = pickle.load(open(fname, 'rb'))
    return D

def main_analysis(D, hack_valid_run=False):
    D['psths'] = dfac.struct({'saccade_onset': dfac.struct(),
    'run_onset': dfac.struct(),
    'grat_onset': dfac.struct(),
    'grat_tuning': dfac.struct()})

    #%% plot average running speed aligned to saccade onset
    from scipy.signal import savgol_filter

    time_bins = np.arange(-4, 4, .01)

    run_time = D.run_data.run_time
    run_spd = np.maximum(D.run_data.run_spd, 0)
    eye_time = D.eye_data.eye_time
    pupil = D.eye_data.pupil


    sac_onsets = D.saccades['start_time']
    sac_onsets = sac_onsets[np.logical_and(sac_onsets > run_time[0] + .5, sac_onsets < run_time[-1] - .5)]
    breaks = np.where(np.diff(sac_onsets)>5)[0]

    plt.figure()
    plt.plot(sac_onsets, np.ones(len(sac_onsets)), 'o')
    val_ix = np.ones(len(run_time), dtype=bool)
    for i in range(len(breaks)):
        plt.axvline(sac_onsets[breaks[i]], color='k', linestyle='--')
        plt.axvline(sac_onsets[breaks[i]+1], color='r', linestyle='--')
        ix = np.logical_and(run_time >= sac_onsets[breaks[i]], run_time <= sac_onsets[breaks[i]+1])
        val_ix[ix] = False

    if hack_valid_run:
        good = val_ix #
    else:
        good = ~np.isnan(run_time)

    plt.plot(run_time, run_spd/np.max(run_spd), 'k')
    plt.plot(run_time, good)

    rspd = dfac.psth_interp(run_time[good], run_spd[good], sac_onsets, time_bins)
    m = np.nanmean(rspd, axis=1)
    sd = np.nanstd(rspd, axis=1) / np.sqrt(rspd.shape[1])
    plt.figure()

    f = plt.fill_between(time_bins, m-sd*2, m+sd*2, color='k', alpha=.5)
    plt.plot(time_bins, m, 'k')
    # val_ix = D.run_data.run_time > 0
    plt.axhline(np.mean(run_spd[good]), color='k', linestyle='--')
    plt.xlabel('Saccade Onset (sec)')
    plt.ylabel('Running Speed (cm/s)')

    D.psths.saccade_onset['run_spd'] = dfac.struct({'mean': m, 'std_error': sd, 'time_bins': time_bins})

    #%% plot histogram of saccade onsets around running onset
    run_onsets = D.run_epochs.start_time
    time_bins = np.arange(-1, 2.5, .02)
    sachist = dfac.psth([D.saccades['start_time']], run_onsets, time_bins)
    plt.figure()
    plt.fill_between(time_bins, np.zeros(len(time_bins)), np.sum(sachist, axis=1).flatten(), color='k', alpha=.5)
    plt.plot(time_bins, np.sum(sachist, axis=1), color='k')
    plt.xlabel("Time from Running Onset (s)")
    plt.ylabel("Saccade Count")

    D.psths.run_onset['saccade_onset'] = dfac.struct({'hist': np.sum(sachist, axis=1).flatten(), 'time_bins': time_bins})

    #%% plot pupil area aligned to saccade onset and running onset

    pupsac = dfac.psth_interp(eye_time, pupil, sac_onsets, time_bins)
    puprun = dfac.psth_interp(eye_time, pupil, run_onsets, time_bins)
    runrun = dfac.psth_interp(run_time[good], run_spd[good], run_onsets, time_bins)

    # pupil area aligned to saccade onset
    plt.figure()
    plt.subplot(1,2,1)
    m = np.nanmean(pupsac, axis=1)
    s = np.nanstd(pupsac, axis=1) / np.sqrt(pupsac.shape[1])
    D.psths.saccade_onset['pupil'] = dfac.struct({'mean': m, 'std_error': s, 'time_bins': time_bins})

    plt.fill_between(time_bins, m-s*2, m+s*2, color='k', alpha=.5)
    plt.plot(time_bins, m, 'k')
    plt.xlabel('Time from Saccade Onset (s)')
    plt.ylabel('Pupil Area (?)')

    plt.subplot(1,2,2)
    m = np.nanmean(puprun, axis=1)
    s = np.nanstd(puprun, axis=1) / np.sqrt(puprun.shape[1])
    plt.fill_between(time_bins, m-s*2, m+s*2, color='k', alpha=.5)
    plt.plot(time_bins, m, 'k')
    plt.xlabel('Time from Running Onset (s)')
    plt.ylabel('Pupil Area (?)')

    D.psths.run_onset['pupil'] = dfac.struct({'mean': m, 'std_error': s, 'time_bins': time_bins})

    #%% control: running speed as a function of runing onset
    plt.figure()

    m = np.nanmean(runrun, axis=1)
    s = np.nanstd(runrun, axis=1) / np.sqrt(runrun.shape[1])
    plt.fill_between(time_bins, m-s*2, m+s*2, color='k', alpha=.5)
    plt.plot(time_bins, m, 'k')
    plt.xlabel('Time from Running Onset (s)')
    plt.ylabel('Running Speed (cm/s)')

    D.psths.run_onset['run_spd'] = dfac.struct({'mean': m, 'std_error': s, 'time_bins': time_bins})

    #%% Analyze spikes!
    from scipy.signal import savgol_filter

    idx = dfac.get_valid_time_idx(D.saccades['start_time'], D.gratings['start_time'].to_numpy(), D.gratings['stop_time'].to_numpy())

    units = D.units[D.units["ecephys_structure_acronym"] == 'VISp']
    # units = D.units
    spike_times = [D.spikes[id] for id in units.index.values]

    bin_size = 0.001
    time_bins = np.arange(-.5, .5, bin_size)


    plt.figure(figsize=(10,3))
    Robs = dfac.psth(spike_times, D.gratings['start_time'].to_numpy(), time_bins)
    m = np.mean(Robs, axis=1).squeeze()/bin_size
    sd = np.std(Robs, axis=1).squeeze()/bin_size / np.sqrt(Robs.shape[1])
    plt.subplot(1,3,1)
    f = plt.plot(time_bins, savgol_filter(m, 101, 3, axis=0))
    plt.title("Grating")
    plt.ylabel("Firing Rate (sp / s)")
    plt.xlabel("Time from Onset (s)")

    D.psths.grat_onset['spikes'] = dfac.struct({'mean': m, 'std_error': sd, 'time_bins': time_bins})


    Robs = dfac.psth(spike_times, D.saccades['start_time'], time_bins)
    m = np.mean(Robs, axis=1).squeeze()/bin_size
    sd = np.std(Robs, axis=1).squeeze()/bin_size / np.sqrt(Robs.shape[1])
    plt.subplot(1,3,2)
    f = plt.plot(time_bins, savgol_filter(m, 101, 3, axis=0))
    plt.title("Saccade")
    plt.xlabel("Time from Onset (s)")

    D.psths.saccade_onset['spikes'] = dfac.struct({'mean': m, 'std_error': sd, 'time_bins': time_bins})

    Robs = dfac.psth(spike_times, D.run_epochs.start_time, time_bins)
    m = np.mean(Robs, axis=1).squeeze()/bin_size
    sd = np.std(Robs, axis=1).squeeze()/bin_size / np.sqrt(Robs.shape[1])
    plt.subplot(1,3,3)
    f = plt.plot(time_bins, savgol_filter(m, 101, 3, axis=0))
    plt.title("Running")
    plt.xlabel("Time from Onset (s)")

    D.psths.run_onset['spikes'] = dfac.struct({'mean': m, 'std_error': sd, 'time_bins': time_bins})


    #%% PSTH / Tuning Curve analysis
    # %matplotlib ipympl

    bin_size = .02
    t1 = np.max(D.gratings.duration )
    time_bins = np.arange(-.25, t1+.25, bin_size)
    Robs = dfac.psth(spike_times, D.gratings['start_time'].to_numpy(), time_bins)
    run_spd  = dfac.psth_interp(D.run_data.run_time, D.run_data.run_spd, D.gratings['start_time'].to_numpy(), time_bins)
    pupil  = dfac.psth_interp(D.eye_data.eye_time, D.eye_data.pupil, D.gratings['start_time'].to_numpy(), time_bins)
    saccades = dfac.psth([D.saccades['start_time']], D.gratings['start_time'].to_numpy(), time_bins).squeeze()

    eye_dx = savgol_filter(D.eye_data.eye_x, 101, 3, axis=0, deriv=1, delta=1/D.eye_data.fs)
    eye_dy = savgol_filter(D.eye_data.eye_y, 101, 3, axis=0, deriv=1, delta=1/D.eye_data.fs)
    eye_spd = np.hypot(eye_dx, eye_dy)
    eye_vel = dfac.psth_interp(D.eye_data.eye_time, eye_spd, D.gratings['start_time'].to_numpy(), time_bins)

    wrapAt = 180
    bad_ix = (D.gratings.orientation=='null')
    condition = D.gratings.orientation
    condition[bad_ix] = 0.1
    condition = condition % wrapAt
    conds = np.unique(condition)
    Nconds = len(conds)

    #%%
    plt.figure(figsize=(10,3))
    plt.subplot(1,4,1)
    plt.imshow(run_spd.T, aspect='auto')
    plt.subplot(1,4,2)
    plt.imshow(pupil.T, aspect='auto')
    plt.subplot(1,4,3)
    plt.imshow(eye_vel.T, aspect='auto')

    # main boot analysis
    run_mod = dict()

    def nanmean(x):
        return np.mean(x*~np.isnan(x), axis=0)

    stationary_trials = np.where(nanmean(run_spd) < 3)[0]
    num_stat = len(stationary_trials)
    running_trials = np.where(nanmean(run_spd) > 5)[0]
    num_run = len(running_trials)

    num_trials = np.minimum(num_stat, num_run)
    print("using %d trials [%d run, %d stationary]" % (num_trials, num_run, num_stat))

    tstim = np.logical_and(time_bins > 0.05, time_bins < t1 + 0.05)
    tbase = time_bins < 0

    spctrun = Robs[:, running_trials, :]
    spctstat = Robs[:, stationary_trials, :]

    def boot_ci(data, n = None, nboot=500):
        if n is None:
            n = data.shape[1]
        bootix = np.random.randint(0,n-1,size=(nboot,n))
        return np.percentile(np.nanmean(data[bootix,:], axis=1), (2.5, 50, 97.5), axis=0)

    frbaseR = boot_ci(np.mean(spctrun[tbase,:,:], axis=0), n = num_trials, nboot=500)/bin_size
    frbaseS = boot_ci(np.mean(spctstat[tbase,:,:], axis=0), n = num_trials, nboot=500)/bin_size
    frstimR = boot_ci(np.mean(spctrun[tstim,:,:], axis=0), n = num_trials, nboot=500)/bin_size
    frstimS = boot_ci(np.mean(spctstat[tstim,:,:], axis=0), n = num_trials, nboot=500)/bin_size

    plt.figure()
    plt.subplot(1,2,1)
    plt.errorbar(frbaseS[1,:], frbaseR[1,:], yerr=np.abs(frbaseR[(0,2),:] - frbaseR[1,:]),
        xerr=np.abs(frbaseS[(0,2),:] - frbaseS[1,:]), fmt='o')
    plt.plot(plt.xlim(), plt.xlim(), 'k--')
    plt.xlabel('Stationary')
    plt.ylabel('Running')

    plt.subplot(1,2,2)
    plt.errorbar(frstimS[1,:], frstimR[1,:], yerr=np.abs(frstimR[(0,2),:] - frstimR[1,:]),
        xerr=np.abs(frstimS[(0,2),:] - frstimS[1,:]), fmt='o')
    plt.plot(plt.xlim(), plt.xlim(), 'k--')
    plt.xlabel('Stationary')
    plt.ylabel('Running')

    D['firing_rate'] = dfac.struct({'base_rate_stat': frbaseS, 'base_rate_run': frbaseR,
        'stim_rate_run': frstimR, 'stim_rate_stat': frstimS})

    #%% analyze tuning curves

    nboot = 500
    def bootstrap_ci(data, n = None, nboot=500):
        if n is None:
            n = data.shape[1]
        bootix = np.random.randint(0,n-1,size=(nboot,n))
        return np.percentile(np.nanmean(data[:,bootix,:], axis=2), (15.75, 84.25), axis=1)


    NC = Robs.shape[-1]
    mu_rate = dfac.struct({'all':np.zeros((Nconds, len(time_bins), NC)),
        'running': np.zeros((Nconds, len(time_bins), NC)),
        'stationary': np.zeros((Nconds, len(time_bins), NC)),
        'running_ns': np.zeros((Nconds, len(time_bins), NC)),
        'stationary_ns': np.zeros((Nconds, len(time_bins), NC)),
        'pupil_large': np.zeros((Nconds, len(time_bins), NC)),
        'pupil_small': np.zeros((Nconds, len(time_bins), NC))})

    n = dict()
    ci_rate = dict()
    for key in mu_rate.keys():
        n[key] = np.zeros(Nconds)
        ci_rate[key] = np.zeros((Nconds, 2, len(time_bins), NC))


    for icond, cond in enumerate(conds):
        
        # all trials
        idx = np.logical_and(condition == cond, D.gratings.duration > .6)
        idx = np.where(idx)[0]
        mu_rate['all'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['all'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['all'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(run_spd, axis=0) > 5)
        mu_rate['running'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['running'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['running'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(run_spd, axis=0) < 3)
        mu_rate['stationary'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['stationary'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['stationary'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(pupil, axis=0) > np.nanmedian(pupil))
        mu_rate['pupil_large'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['pupil_large'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['pupil_large'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(pupil, axis=0) < np.nanmedian(pupil))
        mu_rate['pupil_small'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['pupil_small'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['pupil_small'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(run_spd, axis=0) > 5)
        # stat_time = np.expand_dims(eye_vel[:,idx]<25, axis=2)
        # mu_rate['running_ns'][icond, :, :] = np.mean(Robs[:, idx, :]*stat_time, axis=1)
        
        # idx = np.logical_and(idx, np.sum(saccades, axis=0) == 0)
        idx = np.logical_and(idx, np.sum(eye_vel > 50, axis=0) == 0)
        mu_rate['running_ns'][icond, :, :] = np.mean(Robs[:, idx, :], axis=1)
        n['running_ns'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.mean(run_spd, axis=0) < 3)
        # stat_time = np.expand_dims(eye_vel[:,idx]<25, axis=2)
        # mu_rate['stationary_ns'][icond, :, :] = np.mean(Robs[:, idx, :]*stat_time, axis=1)

        idx = np.logical_and(idx, np.sum(eye_vel > 50, axis=0) == 0)
        mu_rate['stationary_ns'][icond, :, :] = np.mean(Robs[:, idx, :], axis=1)

        n['stationary_ns'][icond] = np.sum(idx)
        
    D.psths.grat_tuning = dfac.struct({'mu_rate': mu_rate, 'n': n, 'ci_rate': ci_rate, 'time_bins': time_bins})
    return D

#%%
# win = 200
# run_time = D.run_data.run_time
# run_spd = D.run_data.run_spd
# run_epochs = dfac.get_run_epochs(run_time, run_spd, win=win)
#%%
%matplotlib inline
D = main_analysis(D)

#%%
save_analyses(D, fpath='./analyses/')



#%%
#%%
#%% load huklab dataset
import data_factory as dfac
sessions = dfac.get_huklab_sessions()

num_sessions = len(sessions[0])
%matplotlib inline

for i in range(3,num_sessions):
    session = dfac.get_huklab_session(sessions[1], sessions[0], i)

    try:
        D = dfac.process_huklab_session(session)

        D = main_analysis(D, hack_valid_run=True)
        save_analyses(D, fpath='./analyses/')
    except:
        print("ERROR on session %d" %i)
        pass



#%% check main sequence

amp = np.hypot(D.saccades['dx'], D.saccades['dy'])
vel = D.saccades['peak_vel']
plt.figure()
plt.plot(amp, vel, '.')
plt.xlabel('Saccade Amplitude (deg)')
plt.ylabel('Saccade Velocity (deg/s)')

#%%
mu_rate = D.psths.grat_tuning.mu_rate
ci_rate = D.psths.grat_tuning.ci_rate
n = D.psths.grat_tuning.n

from scipy.signal import savgol_filter
def get_ylim(x,y):
    mx = np.nanmax((x, y))
    mn = np.nanmin((x, y))
    return mn, mx

smfun = lambda x: savgol_filter(x, 11, 3, axis=0)/savgol_filter(np.ones(len(x)), 11, 3, axis=0)/bin_size

cc = 0
plt.figure(figsize=(10,3))

cond1 = 'running'
cond2 = 'stationary'
cond3 = 'all'
cond4 = 'all'

cidx = np.where(n['all']>0)[0]
# cidx = np.intersect1d(np.where(n[cond1]>0)[0],np.where(n[cond2]>0)[0])
lines_cond1=[]
lines_cond2=[]
lines_cond3=[]
lines_cond4=[]
ax = []

cmap = plt.cm.winter(np.linspace(0,1,len(cidx)))

for i,c in enumerate(cidx):
    ax.append(plt.subplot(1, len(cidx), i+1))
    lines_cond3.append(plt.fill_between(time_bins, smfun(ci_rate[cond1][c,0,:,cc]),
         smfun(ci_rate[cond1][c,1,:,cc]), color=cmap[i,:], alpha=.25))
    lines_cond4.append(plt.fill_between(time_bins, smfun(ci_rate[cond2][c,0,:,cc]),
         smfun(ci_rate[cond2][c,1,:,cc]), color=cmap[i,:]*(1,1,1,.5), alpha=.1))
    
    lines_cond1.append(plt.plot(time_bins, smfun(mu_rate[cond1][c,:,cc].T), color=cmap[i])[0])
    lines_cond2.append(plt.plot(time_bins, smfun(mu_rate[cond2][c,:,cc].T), color=cmap[i]*(1,1,1,.5))[0])
    ax[i].set_xlim((time_bins[3], time_bins[-3]))
    
    ax[i].set_ylim(get_ylim(mu_rate[cond1][:,:,cc]/bin_size, mu_rate[cond2][:,:,cc]/bin_size) )
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    if i>0:
        ax[i].set_yticklabels([])

#%%

def update_fill_plot(collection, y1, y2):
    path = collection.get_paths()
    path[0].vertices[range(0,len(y1)),1] = y2
    path[0].vertices[range(len(y1), len(y1)+len(y2)),1] = y1

cc += 1
if cc >= NC:
    cc = 0

for cond in range(len(cidx)):
    lines_cond1[cond].set_ydata(smfun(mu_rate[cond1][cidx[cond],:,cc].T))
    lines_cond2[cond].set_ydata(smfun(mu_rate[cond2][cidx[cond],:,cc].T))
    
    update_fill_plot(lines_cond3[cond], smfun(ci_rate[cond1][cidx[cond],0,:,cc]), smfun(ci_rate[cond1][cidx[cond],1,:,cc]))
    update_fill_plot(lines_cond4[cond], smfun(ci_rate[cond2][cidx[cond],0,:,cc]), smfun(ci_rate[cond2][cidx[cond],1,:,cc]))
    
    ax[cond].set_ylim(get_ylim(mu_rate[cond1][:,:,cc]/bin_size, mu_rate[cond2][:,:,cc]/bin_size) )

plt.title(cc)


#%% summary plots



def get_mu_ci(data):
    mu = data[1,:]
    ci = data[(0,2),:] - mu
    return mu, ci

def get_psth_rate(data):
    if 'mean' in data.keys():
        mu = data.mean
    else:
        mu = data.hist
    bins = data.time_bins

    return mu, bins

def get_subj_frate(subj, fpath = './analyses/'):
    flist = [f for f in os.listdir(fpath) if subj in f]

    frate = {'base_rate_stat': np.empty(0), 'base_rate_run': np.empty(0), 'stim_rate_run': np.empty(0), 'stim_rate_stat': np.empty(0)}
    frateci = {'base_rate_stat': np.empty(0), 'base_rate_run': np.empty(0), 'stim_rate_run': np.empty(0), 'stim_rate_stat': np.empty(0)}

    for f in flist:
        D = load_analyses(f.split('.')[0])

        for k in D.firing_rate.keys():
            mu,ci = get_mu_ci(D.firing_rate[k])
            frate[k] = np.append(frate[k], mu)
            frateci[k] = np.append(frateci[k], ci)
    return frate, frateci

def get_subj_psth(subj, alignment='run_onset', measured='run_spd', fpath = './analyses/'):
    flist = [f for f in os.listdir(fpath) if subj in f]

    mean_rate = []
    time_bins = []
    
    for f in flist:
        D = load_analyses(f.split('.')[0])

        mu, t = get_psth_rate(D.psths[alignment][measured])
        mean_rate.append(mu)
        time_bins.append(t)
    
    if len(mean_rate[0].shape) == 1:
        m = np.asarray(mean_rate).T
    else:
        m = np.empty((mean_rate[0].shape[0],0))
        for i in range(len(mean_rate)):
            m = np.append(m, mean_rate[i], axis=1)

    return m, time_bins[0]


frateGru, frateCiGru = get_subj_frate('gru')
frateBrie, frateCiBrie = get_subj_frate('brie')
frateMouse, frateCiMouse = get_subj_frate('mouse')

plt.figure()
plt.plot(frateGru['base_rate_stat'], frateGru['base_rate_run'], 'o')
plt.plot(frateBrie['base_rate_stat'], frateBrie['base_rate_run'], 'o')
plt.plot(frateMouse['base_rate_stat'], frateMouse['base_rate_run'], 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Stationary')
plt.ylabel('Running')
plt.title("Baseline Firing Rate")


plt.figure()
plt.plot(frateGru['stim_rate_stat'], frateGru['stim_rate_run'], 'o')
plt.plot(frateBrie['stim_rate_stat'], frateBrie['stim_rate_run'], 'o')
plt.plot(frateMouse['stim_rate_stat'], frateMouse['stim_rate_run'], 'o')
plt.xlabel('Stationary')
plt.ylabel('Running')
plt.title("Stim Firing Rate")
plt.plot(plt.xlim(), plt.xlim(), 'k--')

# %%


D = load_analyses(f.split('.')[0])


# %%

cmap = plt.get_cmap('Set1')
plt.figure()

for i, subj in zip(range(3), ['gru', 'brie', 'mouse']):
    mean_rate, time_bins = get_subj_psth(subj, alignment='run_onset', measured='run_spd')

    plt.plot(time_bins, mean_rate, color=cmap(i))
plt.xlabel('Time from running onset (s)')
plt.ylabel('Running Speed (cm/s)')

#%% run-aligned saccade onset
plt.figure()
from scipy.signal import savgol_filter
for i, subj in zip(range(3), ['gru', 'brie', 'mouse']):
    mean_rate, time_bins = get_subj_psth(subj, alignment='run_onset', measured='saccade_onset')
    mean_rate = savgol_filter(np.mean(mean_rate, axis=1), 11, 3)
    plt.fill_between(time_bins, np.zeros(len(mean_rate)), mean_rate, color=cmap(i), alpha=0.25)
    plt.plot(time_bins, mean_rate, color=cmap(i))

plt.xlabel('Time from running onset (s)')
plt.ylabel('Saccade_count')

#%% pupil aligned to running onset
plt.figure()
from scipy.signal import savgol_filter
for i, subj in zip(range(3), ['gru', 'brie', 'mouse']):
    mean_rate, time_bins = get_subj_psth(subj, alignment='run_onset', measured='pupil')
    if mean_rate[0][0] > 1000:
        for j in range(len(mean_rate)):
            mean_rate[j]/=1000000
    se = savgol_filter(np.std(mean_rate, axis=1), 11, 3) / np.sqrt(mean_rate.shape[1])
    mean_rate = savgol_filter(np.mean(mean_rate, axis=1), 11, 3)
    
    
    # plt.fill_between(time_bins, np.zeros(len(mean_rate)), mean_rate, color=cmap(i), alpha=0.25)
    plt.fill_between(time_bins, mean_rate-se, mean_rate+se, color=cmap(i), alpha=0.25)
    plt.plot(time_bins, mean_rate, color=cmap(i))

plt.xlabel('Time from running onset (s)')
plt.ylabel('Pupil Area (a.u.)')

#%%
plt.figure()
from scipy.signal import savgol_filter
for i, subj in zip(range(3), ['gru', 'brie', 'mouse']):
    mean_rate, time_bins = get_subj_psth(subj, alignment='saccade_onset', measured='pupil')
    if mean_rate[0][0] > 1000:
        for j in range(len(mean_rate)):
            mean_rate[j]/=1000000
    se = savgol_filter(np.std(mean_rate, axis=1), 11, 3) / np.sqrt(mean_rate.shape[1])
    mean_rate = savgol_filter(np.mean(mean_rate, axis=1), 11, 3)
    
    
    # plt.fill_between(time_bins, np.zeros(len(mean_rate)), mean_rate, color=cmap(i), alpha=0.25)
    plt.fill_between(time_bins, mean_rate-se, mean_rate+se, color=cmap(i), alpha=0.25)
    plt.plot(time_bins, mean_rate, color=cmap(i))

plt.xlabel('Time from running onset (s)')
plt.ylabel('Pupil Area (a.u.)')

#%%
use = 'grat'

plt.figure()

for i, subj in zip(range(3), ['gru', 'brie', 'mouse']):

    mean_rateS, time_bins = get_subj_psth(subj, alignment='saccade_onset', measured='spikes')
    mean_rateG, time_bins = get_subj_psth(subj, alignment='grat_onset', measured='spikes')
    mean_rateR, time_bins = get_subj_psth(subj, alignment='run_onset', measured='spikes')
    
    mS = np.mean(mean_rateS, axis=0)
    mG = np.mean(mean_rateG, axis=0)
    mR = np.mean(mean_rateR, axis=0)
    mx = np.asarray([np.max( (mS[cc], mG[cc], mR[cc])) for cc in range(len(mS))])

    if use == 'saccade':
        mean_rate = mean_rateS
    elif use == 'grat':
        mean_rate = mean_rateG
    elif use == 'run':
        mean_rate = mean_rateR

    for cc in range(len(mx)):
        mean_rate[:,cc] = (savgol_filter(mean_rate[:,cc], 51,3) / mx[cc])

    m = np.mean(mean_rate, axis=1)
    s = 2*np.std(mean_rate, axis=1) / np.sqrt(mean_rate.shape[1])
    plt.fill_between(time_bins, m-s, m+s, color=cmap(i), alpha=0.25)
    plt.plot(time_bins, m, color=cmap(i))

if use == 'saccade':
    plt.xlabel('Time from saccade onset (s)')
elif use == 'grat':
    plt.xlabel('Time from grating onset (s)')
elif use == 'run':
    plt.xlabel('Time from running onset (s)')

plt.ylabel('Spike Rate (Normalized)')
#%%

# NT = mean_rate.shape[0]
# NC = [m.shape[1] for m in mean_rate]
# NCtot = sum(NC)
# m = np.zeros( (NT, NCtot) )

# for i in range(len(mean_rate)):
#     mean_rate[0].shape[1]




# plt.plot(t, mu)
# %%
