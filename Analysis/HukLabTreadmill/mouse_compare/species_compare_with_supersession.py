#%% use loco workspace in conda

import os, sys
sys.path.insert(0, '/mnt/Data/Repos/')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib ipympl

# %matplotlib inline
# from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# from neureye.models.display import animshow

#%% load huklab dataset
import loco
sessions = loco.get_huklab_sessions()

num_sessions = len(sessions[0])

Ds = []
unit_list = []

num_sessions = 15
print("Importing %d sessions" % num_sessions)
for i in range(num_sessions):
    print("\n\n\n")
    session = loco.get_huklab_session(sessions[1], sessions[0], i)
    fpath, fname = os.path.split(session.filename)
    print("Importing Session %d [%s]" %(i, fname))

    try:
        Ds.append(loco.process_huklab_session(session))
        unit_list.append(np.array(list(Ds[-1].spikes.keys())))
    except:
        print("Session %d [%s] failed" %(i, fname))
        pass

#%%

def main_analysis(Ds, cc, bin_size=0.02, wrapAt = 180):
    from copy import deepcopy
    from loco import nanmean

    sess_idx = np.where(np.array([cc in u for u in unit_list]))[0]

    gratings = pd.concat([Ds[ss].gratings for ss in list(sess_idx)]).reset_index()
    sessnum = []
    run_spd = []
    robs = []
    t1 = max([np.max(Ds[ss].gratings.duration) for ss in list(sess_idx)])
    time_bins = np.arange(-.25, t1+.25, bin_size)

    for ss in sess_idx:
        D = Ds[ss]
        
        sessnum.append( np.ones(len(D.gratings.duration))*ss )
        robs.append(loco.psth([D.spikes[cc]], D.gratings['start_time'].to_numpy(), time_bins)[:,:,0].T)
        run_spd.append(loco.psth_interp(D.run_data.run_time, D.run_data.run_spd, D.gratings['start_time'].to_numpy(), time_bins).T)


    sessnum = np.concatenate(sessnum)
    robs = np.concatenate( tuple(robs), axis=0)
    run_spd = np.concatenate(run_spd)

    rbase = np.sum(robs[:,time_bins < 0], axis=1)

    comb_inds, S = loco.get_super_session_inds(rbase, sessnum)

    gratings = gratings.loc[comb_inds]
    run_spd = run_spd[comb_inds,:]
    robs = robs[comb_inds,:]
    sessnum = sessnum[comb_inds]

    bad_ix = (gratings.orientation=='null')
    condition = gratings.orientation
    condition[bad_ix] = 0.1
    condition = condition % wrapAt
    conds = np.unique(condition)
    Nconds = len(conds)


    stationary_trials = np.where(nanmean(run_spd, axis=1) < 3)[0]
    num_stat = len(stationary_trials)
    running_trials = np.where(nanmean(run_spd, axis=1) > 5)[0]
    num_run = len(running_trials)

    num_trials = np.minimum(num_stat, num_run)
    print("using %d trials [%d run, %d stationary]" % (num_trials, num_run, num_stat))

    tstim = np.logical_and(time_bins > 0.05, time_bins < t1 + 0.05)
    tbase = time_bins < 0

    spctrun = robs[running_trials, :]
    spctstat = robs[stationary_trials, :]

    frbaseR = loco.bootstrap_ci(np.mean(spctrun[:,tbase], axis=1), n = num_trials, nboot=500, boot_dim=0, mean_dim=1, ci=(2.5, 50, 97.5))/bin_size
    frbaseS = loco.bootstrap_ci(np.mean(spctstat[:, tbase], axis=1), n = num_trials, nboot=500, boot_dim=0, mean_dim=1, ci=(2.5, 50, 97.5))/bin_size
    frstimR = loco.bootstrap_ci(np.mean(spctrun[:,tstim], axis=1), n = num_trials, nboot=500, boot_dim=0, mean_dim=1, ci=(2.5, 50, 97.5))/bin_size
    frstimS = loco.bootstrap_ci(np.mean(spctstat[:,tstim], axis=1), n = num_trials, nboot=500, boot_dim=0, mean_dim=1, ci=(2.5, 50, 97.5))/bin_size

    # tuning curve linear regression
    xax = np.unique(gratings.orientation.values)
    isrunning = (nanmean(run_spd,axis=1) > 5)[:,None]
    X = gratings.orientation.values[:,None] == xax
    X = np.concatenate( (isrunning, X), axis=1)
    R = np.mean(robs[:,tstim], axis=1) / bin_size
    lm = loco.linear_regression(X, R, lam=.1)

    out = {'frBaseR': frbaseR, 'frBaseS': frbaseS, 'frStimR': frstimR, 'frStimS': frstimS, 'xax': xax, 'lm': lm}

    return out
#%%

units = np.unique(np.concatenate(unit_list))
NC = len(units)

S = []
for cc in units:    
    S.append(main_analysis(Ds, cc))

#%% plot

def plot_ellipse(ax, xctr, yctr, xwidth, ywidth, color='b', alpha=.25):
    from matplotlib.patches import Ellipse

    ells = []
    for i in range(len(xctr)):
        ells.append(Ellipse( (xctr[i], yctr[i]), xwidth[i], ywidth[i], angle=0, fill=True, alpha=.5))

    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(alpha)
        e.set_facecolor(color)

    ax.set_xlim(np.min(xctr), np.max(xctr))
    ax.set_ylim(np.min(yctr), np.max(yctr))

def plot_ellipse_error(ax, xctr,yctr,xbounds, ybounds, color='b', alpha=.25):
    from matplotlib.patches import Ellipse

    exctr = (xbounds[:,0] + xbounds[:,1])/2
    eyctr = (ybounds[:,0] + ybounds[:,1])/2
    xwidth = xbounds[:,1] - xbounds[:,0]
    ywidth = ybounds[:,1] - ybounds[:,0]

    plot_ellipse(ax, exctr, eyctr, xwidth, ywidth, color=color, alpha=alpha)

frBaseR = np.array([s['frBaseR'] for s in S])
frBaseS = np.array([s['frBaseS'] for s in S])

plt.figure()
ax = plt.gca()
ells = plot_ellipse_error(ax, frBaseS[:,1], frBaseR[:,1], frBaseS[:,(0,2)], frBaseR[:,(0,2)], color=np.array([1,0,0]), alpha=.25)
plt.plot(frBaseS[:,1], frBaseR[:,1], 'o', color=np.array([1,0,0]), markersize=3)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Stationary')
plt.ylabel('Running')
plt.show()

#%%
# plt.figure()
# plt.subplot(1,2,1)
# plt.errorbar(frbaseS[1,:], frbaseR[1,:], yerr=np.abs(frbaseR[(0,2),:] - frbaseR[1,:]),
#     xerr=np.abs(frbaseS[(0,2),:] - frbaseS[1,:]), fmt='o')
# plt.plot(plt.xlim(), plt.xlim(), 'k--')
# plt.xlabel('Stationary')
# plt.ylabel('Running')

# plt.subplot(1,2,2)
# plt.errorbar(frstimS[1,:], frstimR[1,:], yerr=np.abs(frstimR[(0,2),:] - frstimR[1,:]),
#     xerr=np.abs(frstimS[(0,2),:] - frstimS[1,:]), fmt='o')


#%%
plt.figure()
rbase = np.sum(robs[:,time_bins < 0], axis=1)
trial = np.arange(0, len(rbase))
rateci = np.zeros( (len(sess_idx),3) )
nPotSess = len(sess_idx)
for ss, ii in zip(sess_idx, range(nPotSess)):
    rateci[ii,:] = loco.bootstrap_ci(rbase[sessnum==ss], boot_dim=0, mean_dim=0, ci=(.025, .5, .975))

    plt.plot(trial[sessnum==ss], rbase[sessnum==ss], 'o')

plt.show()

plt.figure()
for ss, ii in zip(sess_idx, range(nPotSess)):
    plt.hist(rbase[sessnum==ss], bins=50, alpha=.5)
plt.show()

#%%
from scipy import stats
ratecompare = np.zeros( (nPotSess, nPotSess))
nTrials = np.zeros(nPotSess)
for i in range(nPotSess):
    nTrials[i] = np.sum(sessnum==sess_idx[i])
    for j in range(nPotSess):
        r = stats.ranksums(rbase[sessnum==sess_idx[i]], rbase[sessnum==sess_idx[j]])
        ratecompare[i,j] = r.pvalue

thresh = 0.01

# find the session combinations with the most Trials
n_tot_trials = np.zeros(nPotSess)
for i in range(nPotSess):
    iix = ratecompare[i,:] > thresh
    n_tot_trials[i] = np.sum(nTrials[iix])


inds = np.argmax(n_tot_trials)
if isinstance(inds, np.int64):
    baseSession = inds
else:
    base_session = inds[0]

iix = ratecompare[baseSession,:] > thresh

comb_inds = []
for ss in sess_idx[iix]:
    comb_inds.append(np.where(sessnum==ss)[0])

np.array(comb_inds)
    
comb_inds = np.array(comb_inds).flatten()


# lm = loco.linear_regression(trial, rbase)
# lm['Probabilities'][1]

# stats.wilcoxon(rbase[sessnum==0], rbase[sessnum==1])
# stats.ranksums(rbase[sessnum==0], rbase[sessnum==1])

# loco.bootstrap_ci(rbase[sessnum==0], boot_dim=0, mean_dim=0, ci=(.025, .975))

#%% debugging sketchpad



#%%
D.spikes.keys()

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
D = Ds[0]
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


#%%
# win = 200
# run_time = D.run_data.run_time
# run_spd = D.run_data.run_spd
# run_epochs = dfac.get_run_epochs(run_time, run_spd, win=win)
#%%
%matplotlib inline
D = loco.main_analysis(D)

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
