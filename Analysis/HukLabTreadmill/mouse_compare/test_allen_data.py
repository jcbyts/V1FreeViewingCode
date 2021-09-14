#%% use loco workspace in conda

import os, sys
import matplotlib
sys.path.insert(0, '/mnt/Data/Repos/')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d
# %matplotlib ipympl
%matplotlib inline
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from neureye.models.display import animshow

#%% load dataset

import data_factory as dfac
sessions, cache = dfac.get_allen_sessions()
session = dfac.get_allen_session(cache, sessions, 40)
D = dfac.process_allen_dataset(session)


#%%
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
ax2.set_ylim((0, .005))

ax3 = plt.subplot(3,1,3)
plt.plot(D.run_data['run_time'], D.run_data['run_spd'], color='k')
ax3.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))
ax3.set_ylim((0, 45))

#%%
igrat = igrat + 10
for ax in [ax1, ax2, ax3]:
    ax.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))

#%% Make animation of it
from matplotlib import animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(10,10))
ax = plt.axes(xlim=(0, 10), ylim=(-2, 2))
igrat = 0

ax1 = plt.subplot(3,1,1)
plt.plot(D.eye_data['eye_time'], D.eye_data['eye_y'], color='k')

nsac = len(D.saccades['start_time'])
for ii in range(nsac):
    iix = np.arange(D.saccades['start_index'][ii], D.saccades['stop_index'][ii], 1, dtype=int)
    plt.plot(D.eye_data['eye_time'][iix], D.eye_data['eye_y'][iix], color='r')

ax1.set_xlim( (-1 + onsets[igrat], onsets[igrat] + 60))
ax1.set_ylim( (-10, 10))
ax1.set_ylabel('Eye Y (deg)')

ax2 = plt.subplot(3,1,2)
plt.plot(D.eye_data['eye_time'], D.eye_data['pupil'], color='k')
for i in range(len(D.saccades['start_time'])):
    plt.axvline(D.saccades['start_time'][i], color='r', linestyle='-')
ax2.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))
ax2.set_ylim((0, .005))
ax2.set_ylabel('Pupil Area')

ax3 = plt.subplot(3,1,3)
line = plt.plot(D.run_data['run_time'], D.run_data['run_spd'], color='k')[0]
for i in range(len(D.saccades['start_time'])):
    plt.axvline(D.saccades['start_time'][i], color='r', linestyle='-')
ax3.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))
ax3.set_ylim((0, 55))
ax3.set_xlabel('Seconds')
ax3.set_ylabel('Speed (cm/s)')
# line, = ax.plot([], [], lw=2)

def init():
    line.set_data(D.run_data['run_time'], D.run_data['run_spd'])
    return line,

# animation function.  This is called sequentially
def animate(i):
    ax1.set_xlim( (0 + onsets[i], onsets[i] + 60))
    ax2.set_xlim( (0 + onsets[i], onsets[i] + 60))
    ax3.set_xlim( (0 + onsets[i], onsets[i] + 60))
    line.set_data(D.run_data['run_time'], D.run_data['run_spd'])
    return line,
    
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=range(400), interval=1, blit=True, repeat=False)

#%%
anim.save('running_data2.mp4', fps=10)

#%%
# First set up the figure, the axis, and the plot element we want to animate
from IPython.display import HTML
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    ax.set_title(str(i))
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=range(400), interval=20, blit=True,repeat=False)
anim.save('basic_animation.mp4', fps=30)
# HTML(anim.to_html5_video())
#%%

# from scipy.signal import interp1d

onsets = np.where(np.diff( (D.run_data.run_spd > 3).astype(float)) == 1)[0]
onsets = D.run_data.run_time[onsets]

#%%

#%% check main sequence

amp = np.hypot(D.saccades['dx'], D.saccades['dy'])
vel = D.saccades['peak_vel']
plt.figure()
plt.plot(amp, vel, '.')
plt.xlabel('Saccade Amplitude (deg)')
plt.ylabel('Saccade Velocity (deg/s)')


#%%
def psth_interp(time, data, onsets, time_bins):
    NT = len(time_bins)
    NStim = len(onsets)
    rspd = np.zeros((NT, NStim))
    fint = interp1d(time, data, kind='linear')

    for istim in range(NStim):
        rspd[:,istim] = fint(time_bins + onsets[istim])
    
    return rspd

# import interp1d from scipy
def psth(spike_times, onsets, time_bins):
    bin_size = np.mean(np.diff(time_bins))
    NC = len(spike_times)
    NStim = len(onsets)
    NT = len(time_bins)
    Robs = np.zeros((NT, NStim, NC))
    for istim in range(NStim):
        for iunit in range(NC):
            st = spike_times[iunit]-onsets[istim]
            st = st[np.logical_and(st >= time_bins[0], st <= (time_bins[-1]+bin_size))]
            Robs[np.digitize(st, time_bins)-1,istim, iunit] += 1

    return Robs

def get_valid_time_idx(times, valid_start, valid_stop):
    valid = np.zeros(len(times), dtype=bool)
    for t in range(len(valid_start)):
        valid = np.logical_or(valid, np.logical_and(times >= valid_start[t], times <= valid_stop[t]))
    return valid

#%%
from scipy.signal import savgol_filter
time_bins = np.arange(-2, 2, .01)
run_time = D.run_data.run_time.to_numpy()
sac_onsets = D.saccades['start_time']
sac_onsets = sac_onsets[np.logical_and(sac_onsets > run_time[0] + .5, sac_onsets < run_time[-1] - .5)]
rspd = psth_interp(run_time, D.run_data.run_spd, sac_onsets, time_bins)
m = np.mean(rspd, axis=1)
sd = np.std(rspd, axis=1) / np.sqrt(rspd.shape[1]) * 2
plt.figure()
f = plt.errorbar(time_bins, m, sd)
plt.xlabel('Saccade Onset (sec)')
plt.ylabel('Running Speed (cm/s)')

#%%
idx = get_valid_time_idx(D.saccades['start_time'], D.gratings['start_time'].to_numpy(), D.gratings['stop_time'].to_numpy())

units = D.units[D.units["ecephys_structure_acronym"] == 'VISp']
# units = D.units
spike_times = [D.spikes[id] for id in units.index.values]

time_bins = np.arange(-.5, .5, .001)
Robs = psth(spike_times, D.saccades['start_time'][idx], time_bins)
# Robs = psth(spike_times, D.gratings['start_time'].to_numpy(), time_bins)

m = np.mean(Robs, axis=1).squeeze()
from scipy.signal import savgol_filter
plt.figure()
f = plt.plot(time_bins, savgol_filter(m, 101, 3, axis=0))

#%%
time_bins = np.arange(-2.5, 2.5, .01)
sachist = psth([D.saccades['start_time']], onsets.to_numpy(), time_bins)
plt.figure()
plt.plot(time_bins, np.mean(sachist, axis=1))
#%%
plt.figure()
# m = savgol_filter(m, 101, 3, axis=0)
plt.imshow(m - np.mean(m, axis=0), aspect='auto')


#%%
from scipy.interpolate import interp1d

time_bins = session.running_speed['start_time'].values
NC = len(units.index.values)
stim_dur = np.max(presentations.duration.values)
time_bins = np.arange(-.5, stim_dur + .5, .01)
NT = len(time_bins)
NStim = len(presentations)

Robs = np.zeros((NT, NStim, NC))
rspd = np.zeros((NT, NStim))
pup = np.zeros((NT, NStim))

fpup = interp1d(D.eye_data.eye_time, D.eye_data.pupil, kind='linear')
frun = interp1d(D.run_data.run_time, D.run_data.run_spd, kind='linear')

for istim in range(NStim):
    pup[:,istim] = fpup(time_bins + onset[istim])
    rspd[:,istim] = frun(time_bins + onset[istim])
    for iunit, unit in zip(range(NC), units.index.values):
        st = session.spike_times[unit]-onset[istim]
        st = st[np.logical_and(st > -.5, st < stim_dur + .5)]
        Robs[np.digitize(st, time_bins)-1,istim, iunit] += 1

#%%
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
# plt.plot(run_time, run_spd)
# plt.plot(onset, np.zeros(NStim), 'o')
plt.imshow(rspd.T, aspect='auto', cmap='viridis')

plt.subplot(1,3,2)
plt.imshow(pup.T, aspect='auto', cmap='viridis')

plt.subplot(1,3,3)
plt.imshow(np.mean(Robs, axis=2).T, aspect='auto', cmap='viridis')


#%%
# plt.plot(np.mean(Robs, axis=1), running_speed.velocity.values/100, '.')
plt.figure()
plt.plot(eye_time, pupil*10000)
plt.plot(run_time, run_spd)
plt.plot(time_bins, gaussian_filter1d(np.mean(Robs, axis=1), sigma=15)*100 - 10)
plt.show()
# plt.plot(running_speed.velocity.values/100)
# plt.xlim((0, 10000))
# plt.ylim((0, 1))


#%%
plt.figure()
f = plt.plot(pup[:,10:12])
plt.show()

#%%
time_step = 0.01
time_bins = np.arange(-0.1, 0.5 + time_step, time_step)

histograms = session.presentationwise_spike_counts(
    stimulus_presentation_ids=presentations.index.values,  
    bin_edges=time_bins,
    unit_ids=units.index.values
)

histograms.coords

# %%
mean_histograms = histograms.mean(dim="stimulus_presentation_id")

fig, ax = plt.subplots(figsize=(8, 8))
ax.pcolormesh(
    mean_histograms["time_relative_to_stimulus_onset"], 
    np.arange(mean_histograms["unit_id"].size),
    mean_histograms.T, 
    vmin=0,
    vmax=1
)

ax.set_ylabel("unit", fontsize=24)
ax.set_xlabel("time relative to stimulus onset (s)", fontsize=24)
ax.set_title("peristimulus time histograms for VISp units on flash presentations", fontsize=24)

plt.show()
# %%

mean_histograms
# %%
