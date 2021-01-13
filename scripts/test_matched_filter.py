"""
Show creation of an injection and analysis of a matched-filter SNR for a
single template.
"""

import ringdown
import pycbc.filter
import numpy as np
from matplotlib import pyplot as plt


starttime = 1000000000.0  # start time of data
duration = 2.0  # duration of data
dt = 1.0 / 8192.0  # time step

tc = starttime + 0.5  # start time of the ring-down signal (GPS second)
freq = 1234.5  # frequency of signal (Hz)
amp = 5e-21  # amplitude of signal
tau = 0.07  # decay time (seconds)
ra = 0.3  # source right ascension (rads)
dec = -0.3  # source declination (rads)
psi = 1.1  # source polarisation angle (rads)
phi = 0.2  # initial phase (rads)
inclination = 0.9  # source inclination angle (rads)

detector = "H1"

# default PSD will be aLIGO design curve
injH1 = ringdown.RingdownInjections(
    tc,
    freq,
    amp,
    tau,
    ra,
    dec,
    psis=psi,
    phis=phi,
    inclinations=inclination,
    detector=detector,
    starttime=starttime,
    duration=duration,
    deltat=dt,
)

# get signal SNR
stilde_signal = injH1.injection_data.to_frequencyseries()  # pure signal in frequency domain
psdnonzero = np.where(injH1.psd.data[:len(stilde_signal)] != 0)  # indices for non-zero PSD
snr = np.sqrt(
        (4 / duration) * np.sum(np.conj(stilde_signal.data[psdnonzero]) * stilde_signal.data[psdnonzero] / injH1.psd.data[psdnonzero])
).real
print("Signal optimal SNR is {0:.1f}".format(snr))

stilde = injH1.data.to_frequencyseries()

frange = [1230, 1240]  # frequency range (Hz)
taurange = [0.06, 0.07]  # Quality factor ranges
mm = 0.03  # maximum mismatch

tb = ringdown.RingdownTemplateBank(frange, taurange=taurange, mm=mm)
flow = 20.0
for i, waveform in enumerate(tb.generate_waveforms(domain="frequency", deltaf=stilde.delta_f)):
    hp, hc = waveform

    hp.resize(len(stilde))
    snr = pycbc.filter.matched_filter(hp, stilde, psd=injH1.psd,
                                      low_frequency_cutoff=flow)

    plt.plot(snr.sample_times, abs(snr))
    plt.ylabel('signal-to-noise ratio')
    plt.xlabel('time (s)')
    plt.show()

    if i == 0:
        # just run for one loop
        break
