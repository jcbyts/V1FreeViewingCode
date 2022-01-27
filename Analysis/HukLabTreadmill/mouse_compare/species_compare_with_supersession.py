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

fig_dir = '/mnt/Data/Figures/'
import loco

#%% main functions

def load_sessions(dataset, subj='all'):

    Ds = []

    if dataset=='huk':

        sessions = loco.get_huklab_sessions()

        # remove bad sessions
        sesslist = [s for s in sessions[0] if not s=='gru_20210426_grat.mat']
        sessions = (sesslist, sessions[1])

        if not subj == 'all':
            print("Selecting subject %s" %subj)
            sessions_ = [session for session in sessions[0] if session.split('_')[0]==subj]
            sessions = tuple([sessions_, sessions[1]])

        num_sessions = len(sessions[0])

        print("Importing %d sessions" % num_sessions)
        for i in range(num_sessions):
            print("\n\n\n")
            session = loco.get_huklab_session(sessions[1], sessions[0], i)
            fpath, fname = os.path.split(session.filename)
            print("Importing Session %d [%s]" %(i, fname))

            try:
                Ds.append(loco.process_huklab_session(session))
            except:
                print("Session %d [%s] failed" %(i, fname))
                pass
    
    if dataset=='allen':
        data_directory = '/mnt/Data/Datasets/allen/ecephys_cache_dir/'
        sessions = loco.get_allen_sessions(data_directory = data_directory)

        if subj == 'downloaded':
            downloaded_sessions = [f for f in os.listdir(data_directory) if 'session_' in f]

            # sessids = [np.where(int(s.split('_')[1])== sessions[0].index.values)[0][0] for s in downloaded_sessions]
            sessids = [int(s.split('_')[1]) for s in downloaded_sessions]

        elif subj == 'all':
            sessids = sessions[0].index.values
            
        else:
            sessids = sessions[0].index.values[sessions[0].session_type == subj]
        
        num_sessions = len(sessids)
        print("Importing %d sessions based on filter [%s]" %(num_sessions, subj))

        Ds = []
        for i in range(num_sessions):
            print("\n\n\n")
            
            print("Importing Session %d [%d]" %(i, sessids[i]))
            try:
                session = loco.get_allen_session(sessions[1], sessions[0], sessids[i])
                Ds.append(loco.process_allen_dataset(session))
            except:
                print("Session %d [%d] failed" %(i, sessids[i]))
                pass

    return Ds

def analyze_super_session(Ds):
    from tqdm import tqdm

    unit_list = [D.units.index[D.units.ecephys_structure_acronym=='VISp'].values for D in Ds]
    units = np.unique(np.concatenate(unit_list))
    NC = len(units)

    S = []
    print("Analyzing %d units" %NC)
    for cc in tqdm(units):    
        S.append(loco.spike_count_analysis(Ds, cc))
    return S

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

def plot_spike_count(S, cond, trial_thresh=50, color='b', alpha=.25, markersize=3, vis_only=False):
    num_trials = np.array([S[i]['num_trials'] for i in range(len(S))])
    iix = num_trials > trial_thresh
    if vis_only:
        pval = np.array([S[i]['stim_driven_pval'] for i in range(len(S))])
        iix = np.logical_and(iix,pval<0.05)

    if cond=='Base':
        frR = np.array([s['frBaseR'] for s in S])
        frS = np.array([s['frBaseS'] for s in S])

    elif cond=='Stim':
        frR = np.array([S[i]['frStimR'] for i in range(len(S))])
        frS = np.array([S[i]['frStimS'] for i in range(len(S))])
    elif cond=='Max':
        frR = np.array([S[i]['frMaxR'] for i in range(len(S))])
        frS = np.array([S[i]['frMaxS'] for i in range(len(S))])

    frR = frR[iix,:]
    frS = frS[iix,:]

    iix = ~np.logical_or(np.isnan(frR[:,1]), np.isnan(frS[:,1]))
    frR = frR[iix,:]
    frS = frS[iix,:]

    ax = plt.gca()
    ells = plot_ellipse_error(ax, frS[:,1], frR[:,1], frS[:,(0,2)], frR[:,(0,2)], color=color, alpha=alpha)
    plt.plot(frS[:,1], frR[:,1], 'o', color=color, markersize=markersize)

    return frR, frS

def sig_check(frR, frS):
    from scipy import stats
    

    num_inc = np.sum(frR[:,1]>frS[:,2])
    num_sup = np.sum(frR[:,1]<frS[:,0])
    print("%d (%02.2f) neurons have increased firing during running" %(num_inc, num_inc/len(frR)))
    print("%d (%02.2f) neurons have decreased firing during running" %(num_sup, num_sup/len(frR)))
    
    print("Wilcoxon signed rank test:")
    med = np.median(frR[:,1]-frS[:,1])
    res = stats.wilcoxon(frR[:,1], frS[:,1])
    if res.pvalue < .05:
        print("Significant difference in firing rate [med=%2.2f, p=%02.4f]" %(med, res.pvalue))
    else:
        print("No significant difference in firing rate [med=%2.2f, p=%02.4f]" %(med, res.pvalue))
    
    print("T-test on geometric mean ratio:")
    iix = np.logical_and(frS[:,1] > 0, frR[:,1] > 0)
    
    lograt = np.log(frR[iix,1]) - np.log(frS[iix,1])

    ci = loco.bootstrap1d(lograt, np.mean, 1000, len(lograt))
    print("Bootstrapped 95%% CI: [%2.2f, %2.2f]" %(np.exp(ci[0]), np.exp(ci[1])))
    res = stats.ttest_1samp(lograt, 0)
    geomean = np.exp(np.mean(lograt))
    if res.pvalue < .05:
        print("Significant difference in firing rate [geomean=%2.2f, p=%02.4f]" %(geomean, res.pvalue))
    else:
        print("No significant difference in firing rate [geomean=%2.2f, p=%02.4f]" %(geomean, res.pvalue))

def get_unit_rf(Ds, cc):
    
    unit_list = [np.array(list(D.spikes.keys())) for D in Ds]
    units = np.unique(np.concatenate(unit_list))
    assert cc in units, 'cc not in unit_list'

    sess_idx = np.where(np.array([cc in u for u in unit_list]))[0]

    nsess = len(sess_idx)
    
    rfmat = []
    azimuth_rf = np.zeros(nsess)
    elevation_rf = np.zeros(nsess)
    xax = np.nan
    yax = np.nan
    success = False

    for i in range(nsess):
        D = Ds[sess_idx[i]]

        try:
            if np.sum(D.units['rf_mat'][cc]) > 0:
                rfmat.append(D.units['rf_mat'][cc])
                azimuth_rf[i] = D.units['azimuth_rf'][cc]
                elevation_rf[i] = D.units['elevation_rf'][cc]
                xax = D.units['xax'][cc].flatten()
                yax = D.units['yax'][cc].flatten()
                success = True
        except:
            pass
    
    return {'rf': rfmat, 'azimuth_rf': azimuth_rf, 'elevation_rf': elevation_rf, 'xax': xax, 'yax': yax, 'success': success}

def get_all_rfs(Ds):
    from tqdm import tqdm
    unit_list = [np.array(list(D.spikes.keys())) for D in Ds]
    units = np.unique(np.concatenate(unit_list))

    rf = []
    for cc in tqdm(units):
        rf.append(get_unit_rf(Ds, cc))
    return rf

def sessionwise(Ds):
    for i in range(len(Ds)):
        # try:
        print("Analyzing session %d" %i)
        Ds[i] = loco.sessionwise_analysis(Ds[i], hack_valid_run=False, plot_figures=False)
        # except:
        #     print("Session %d failed" %i)
        #     pass
    return Ds

def plot_behavior(Ds, alignment='run_onset', field='run_spd', color='k', normalize=False, smoothing=0):
    from scipy.signal import savgol_filter
    time_bins = [D['psths'][alignment][field]['time_bins'] for D in Ds]
    try:
        run_spd = [D['psths'][alignment][field]['mean'] for D in Ds]
    except:
        run_spd = [D['psths'][alignment][field]['hist'] for D in Ds]

    if smoothing > 0:
        run_spd = [savgol_filter(r, smoothing, 1) for r in run_spd]

    if normalize:
        # run_spd = [spd - np.nanmean(spd) for spd in run_spd]
        run_spd = [spd/np.nanmax(spd) for spd in run_spd]
        
    f = plt.plot(np.asarray(time_bins).T, np.asarray(run_spd).T, color=color,alpha=.25)
    plt.plot(time_bins[0], loco.nanmean(np.asarray(run_spd), axis=0), color=color, linewidth=4)

# Colormap for mouse and marmoset
cmap = plt.cm.tab10(np.arange(10))
cmap[3,:-1] = .25
cmap[1,:] = cmap[0,:]
cmap[0,:] = cmap[3,:]


#%% Load data
Dmouse = load_sessions('allen', subj='functional_connectivity')
#%%
Dmarm1 = load_sessions('huk', subj='gru')
# Dmarm2 = load_sessions('huk', subj='brie')

#%%
# save Dmarm1, Dmarm2, and Dmouse
fpath = '/mnt/Data/Datasets/HuklabTreadmill/'
import pickle
with open(os.path.join(fpath, 'Dmarm1.pkl'), 'wb') as f:
    pickle.dump(Dmarm1, f)
with open(os.path.join(fpath, 'Dmarm2.pkl'), 'wb') as f:
    pickle.dump(Dmarm2, f)
with open(os.path.join(fpath, 'Dmouse.pkl'), 'wb') as f:
    pickle.dump(Dmouse, f)
print("Done")

#%% Load sessions alltogether for plotting
fpath = '/mnt/Data/Datasets/HuklabTreadmill/'
import pickle
with open(os.path.join(fpath, 'Dmarm1.pkl'), 'rb') as f:
    Dmarm1 = pickle.load(f)
with open(os.path.join(fpath, 'Dmarm2.pkl'), 'rb') as f:
    Dmarm2 = pickle.load(f)
with open(os.path.join(fpath, 'Dmouse.pkl'), 'rb') as f:
    Dmouse = pickle.load(f)

#%% run sessionwise analyses
Dmarm1 = sessionwise(Dmarm1)
Dmarm2 = sessionwise(Dmarm2) 
Dmouse = sessionwise(Dmouse)


#%%
from scipy.signal import savgol_filter
from copy import deepcopy


plt.figure()
for i in range(len(Dmouse)):
    
    
    run_spd = savgol_filter(deepcopy(Dmouse[i]['run_data']['run_spd']), 31, 3)
    Dmouse[i]['run_epochs'] = loco.get_run_epochs(Dmouse[i]['run_data']['run_time'], run_spd, debug=False, refrac=1, min_duration=1)
    time_bins = np.linspace(-1, 4, 100)
    plt.plot(Dmouse[i]['eye_data']['pupil'])
    # X = loco.psth_interp(Dmouse[i]['run_data']['run_time'], run_spd, Dmouse[i]['run_epochs']['start_time'], time_bins)
    X = loco.psth_interp(Dmouse[i]['eye_data']['eye_time'], Dmouse[i]['eye_data']['pupil'], Dmouse[i]['run_epochs']['start_time'], time_bins)
    
    # plt.plot(np.nanmean(X, axis=1))

    

    # Dmouse[i]['run_epochs']
    # plt.plot(Dmouse[i]['run_data']['run_time'], Dmouse[i]['run_data']['run_spd'])

plt.show()


#%%

# for i in range(len(Dmarm2)):

#     Dmarm2[i]['run_epochs'] = loco.get_run_epochs(Dmarm2[i]['run_data']['run_time'], Dmarm2[i]['run_data']['run_spd'], debug=False, refrac=5)

#%% plot running speed
import seaborn as sns

# find marmov5 sessions for Marm1
from datetime import datetime
start = datetime.strptime('20210505', '%Y%m%d')
datestr = [D['files']['session'].split('_')[1] for D in Dmarm1]
gooddates = [datetime.strptime(d, '%Y%m%d')>start for d in datestr]
Ds = [Dmarm1[i] for i in np.where(gooddates)[0]]

plt.figure(figsize=(5,4))
plot_behavior(Dmouse, alignment='run_onset', field='run_spd', color=cmap[0,:])
plot_behavior(Ds, alignment='run_onset', field='run_spd', color=cmap[1,:])
plot_behavior(Dmarm2, alignment='run_onset', field='run_spd', color=cmap[2,:])
plt.xlabel('Time from running onset (s)')
plt.ylabel('Running Speed (cm/s)')
sns.despine(trim=True, offset=0)

plt.savefig(os.path.join(fig_dir, 'huklab_loco_sfn_run_speed_overlap.pdf'))

#%% plot pupil area

plt.figure(figsize=(5,4))
plot_behavior(Dmouse, alignment='run_onset', field='pupil', color=cmap[0,:], normalize=True)
plot_behavior(Ds, alignment='run_onset', field='pupil', color=cmap[1,:], normalize=True)
plot_behavior(Dmarm2, alignment='run_onset', field='pupil', color=cmap[2,:], normalize=True)
plt.xlabel('Time from running onset (s)')
plt.ylabel('Pupil Area (normalized by maximum)')
sns.despine(trim=True, offset=0)
plt.savefig(os.path.join(fig_dir, 'huklab_loco_sfn_pupil_overlap.pdf'))

#%%

plt.figure(figsize=(5,4))
plot_behavior(Dmouse, alignment='run_onset', field='saccade_onset', color=cmap[0,:], normalize=False, smoothing=11)
plot_behavior(Ds, alignment='run_onset', field='saccade_onset', color=cmap[1,:], normalize=False, smoothing=11)
plot_behavior(Dmarm2, alignment='run_onset', field='saccade_onset', color=cmap[2,:], normalize=False, smoothing=11)
plt.xlabel('Time from running onset (s)')
plt.ylabel('Saccade Rate')
sns.despine(trim=True, offset=0)
plt.savefig(os.path.join(fig_dir, 'huklab_loco_sfn_saccades_overlap.pdf'))

#%% Example RFs
rfs1 = get_all_rfs(Dmarm1)
# rfs2 = get_all_rfs(Dmarm2)

#%%


%matplotlib inline
rf = rfs1
inds = np.where([r['success'] for r in rf])[0]

# restrict to clean examples
x = np.asarray([np.std(rf[i]['rf'][0]) for i in inds])
inds = inds[x>10]
inds = inds[:-4]

n = len(inds)
plt.figure(figsize=(10,10))
nx = int(np.sqrt(n))
ny = int(np.ceil(n/nx))
for i in range(n):
    plt.subplot(nx, ny, i+1)
    extent = [rf[inds[i]]['xax'][0], rf[inds[i]]['xax'][-1], rf[inds[i]]['yax'][0], rf[inds[i]]['yax'][-1]]
    plt.imshow(rf[inds[i]]['rf'][0].T, origin='lower', cmap='coolwarm', extent=extent)
    # plt.title("%2.0f" %x[i])
    plt.axhline(0, color='k')
    plt.axvline(0, color='k')
    plt.axis('off')

figname = os.path.join(fig_dir, 'huklab_loco_sfn_example_rfs.pdf')
plt.savefig(figname, bbox_inches='tight')

#%%


#%%

plt.figure(figsize=(2.5,2.5))


inds = np.where([r['success'] for r in rfs1])[0]
x = np.asarray([rfs1[i]['azimuth_rf'][0] for i in inds])
y = np.asarray([rfs1[i]['elevation_rf'][0] for i in inds])
iix = x > 0
x = x[iix]
y = y[iix]

plt.plot(x,y,'o', markersize=2, alpha=.5, color=cmap[1,:])

rfs2 = get_all_rfs(Dmarm2)
inds = np.where([r['success'] for r in rfs2])[0]
x = np.asarray([rfs2[i]['azimuth_rf'][0] for i in inds])
y = np.asarray([rfs2[i]['elevation_rf'][0] for i in inds])
iix = x > 0
x = x[iix]
y = y[iix]

plt.plot(x,y,'o', markersize=2, alpha=.5, color=cmap[2,:])
plt.grid('on')
plt.axis('square')
plt.xlim((-5, 5))
plt.ylim((-5, 5))

plt.xlabel('Azimuth')
plt.ylabel('Elevation')

plt.legend(['Marmoset 1', 'Marmoset 2'])
# sns.despine(trim=True, offset=0)


plt.savefig(os.path.join(fig_dir, 'huklab_loco_sfn_rf_locations.pdf'), bbox_inches='tight')

plt.show()



#%%
print("%d Mouse sessions" %len(Dmouse))
print("%d Gru sessions" %len(Dmarm1))
print("%d Brie sessions" %len(Dmarm2))

#%%

Smouse = analyze_super_session(Dmouse)
Smarm1 = analyze_super_session(Dmarm1)
Smarm2 = analyze_super_session(Dmarm2)

#%% save 
print("Saving Analyses...")
with open(os.path.join(fpath, 'Smarm1.pkl'), 'wb') as f:
    pickle.dump(Smarm1, f)
with open(os.path.join(fpath, 'Smarm2.pkl'), 'wb') as f:
    pickle.dump(Smarm2, f)
with open(os.path.join(fpath, 'Smouse.pkl'), 'wb') as f:
    pickle.dump(Smouse, f)
print("Done")

#%% Load sessions alltogether for plotting
with open(os.path.join(fpath, 'Smarm1.pkl'), 'rb') as f:
    Smarm1 = pickle.load(f)
with open(os.path.join(fpath, 'Smarm2.pkl'), 'rb') as f:
    Smarm2 = pickle.load(f)
with open(os.path.join(fpath, 'Smouse.pkl'), 'rb') as f:
    Smouse = pickle.load(f)

#%%
print("%d Mouse units" %len(Smouse))
print("%d Gru units" %len(Smarm1))
print("%d Brie units" %len(Smarm2))
#%%


cond = 'Stim'
vis_only=True

plt.figure(figsize=(5,5))


print("Plotting %s Cond" %cond)
print("Mouse")
frR,frS = plot_spike_count(Smouse, cond, color=cmap[0,:], alpha=.1, markersize=1, trial_thresh=0, vis_only=vis_only)
print("N=%d" %len(frR))
sig_check(frR, frS)

print("Gru")
frR,frS = plot_spike_count(Smarm1, cond, color=cmap[1,:], trial_thresh=0, alpha=.1, markersize=1, vis_only=vis_only)
print("N=%d" %len(frR))
sig_check(frR, frS)

print("Brie")
frR,frS = plot_spike_count(Smarm2, cond, color=cmap[2,:], trial_thresh=0, alpha=.1, markersize=1, vis_only=vis_only)
print("N=%d" %len(frR))
sig_check(frR, frS)

plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Spike Rate (Stationary)')
plt.ylabel('Spike Rate (Running)')
plt.axis('square')

plt.title(cond)

plt.legend(['Mouse', 'Marmoset 1', 'Marmoset 2'])
sns.despine(trim=True, offset=0)

plt.savefig(os.path.join(fig_dir, 'huklab_loco_sfn_spike_rate_overlap_%s_vis_%d.pdf' %(cond, vis_only)), bbox_inches='tight')



#%%
fig_dir = '.'
cond = 'Base'
trialthresh = 0
vis_only=True

plt.figure(figsize=(8,4))
# plt.figure(figsize=(16,8))
xd = 50

print("Plotting %s Cond" %cond)
print("Mouse")
plt.subplot(1,3,1)
frR,frS = plot_spike_count(Smouse, cond, color=cmap[0,:], alpha=.025, markersize=1, trial_thresh=trialthresh, vis_only=vis_only)
sig_check(frR, frS)
plt.axis('square')
plt.xlim(0,xd)
plt.ylim(0,xd)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Stationary')
plt.ylabel('Running')
plt.title("Mouse")
sns.despine(trim=True, offset=5)


plt.subplot(1,3,2)
print("Gru")
frR,frS = plot_spike_count(Smarm1, cond, color=cmap[1,:], trial_thresh=trialthresh, alpha=.05, markersize=1, vis_only=vis_only)
sig_check(frR, frS)

plt.axis('square')
plt.xlim(0,xd)
plt.ylim(0,xd)
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Stationary')
plt.title("Marmoset 1")
sns.despine(trim=True, offset=5)

plt.subplot(1,3,3)
print("Brie")
frR,frS = plot_spike_count(Smarm2, cond, color=cmap[2,:], trial_thresh=trialthresh, alpha=.05, markersize=1, vis_only=vis_only)
sig_check(frR, frS)
plt.axis('square')
plt.xlim(0,xd)
plt.ylim(0,xd)

plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlabel('Stationary')
plt.title("Marmoset 2")
sns.despine(trim=True, offset=5)


plt.savefig(os.path.join(fig_dir, 'huklab_loco_sfn_spike_rate_overlap_%s_vis_%d.pdf' %(cond, vis_only)), bbox_inches='tight')

plt.show()
#%% analyze session by session




#%% load huklab dataset


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
