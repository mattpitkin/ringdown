"""
Show matched filter analysis on a portion of real detector data.
"""

import ringdown
import pycbc.filter
import numpy as np
from gwpy.timeseries import TimeSeries
from pycbc.types.timeseries import TimeSeries as PyCBCTimeSeries
from pycbc.psd import interpolate
from matplotlib import pyplot as plt


# download a 128 second chunk of data (the whole data length will be used to generate the
# power spectral density to get a good average)
duration = 128  # number of seconds of data to download
gpsstart = 1187008884  # start time
gpsend = gpsstart + duration
samplerate = 16384

detector = "L1"

# fetch the open data from GWOSC
data = TimeSeries.fetch_open_data(
    detector,
    gpsstart,
    gpsend,
    sample_rate=samplerate,
    format='hdf5',
    host='https://www.gw-openscience.org',
    verbose=False,
    cache=True,
)

# convert the data to a PyCBC time series
pycbcdata = PyCBCTimeSeries(data.data, delta_t=(1 / data.sample_rate.value))

# high-pass filter the data to only include the frequencies we're interested in
lowcutoff = 1000
buffer = 50  # just allow a bit of a buffer at the edges
pycbcdata = pycbcdata.highpass_fir(lowcutoff - buffer, 8)  # 8 is the "order" of the filter

# create the template bank
frange = [1230, 1240]  # frequency range (Hz)
taurange = [0.06, 0.07]  # Quality factor ranges
mm = 0.03  # maximum mismatch
tb = ringdown.RingdownTemplateBank(frange, taurange=taurange, mm=mm)
flow = lowcutoff

# get the initial chunk of data to matched filter - let's get four seconds
# after ignoring the first two seconds due to the filter response
respbuffer = 2
chunkdur = 4
inidata = pycbcdata.crop(respbuffer, duration - (respbuffer + chunkdur))
initilde = inidata.to_frequencyseries()

# get the psd using the whole data segment (using 128 / 16 length segments)
psd = pycbcdata.filter_psd(128 / 16, (8 / samplerate), flow)
psd = interpolate(psd, initilde.delta_f)

# perform matched filtering
for i, waveform in enumerate(tb.generate_waveforms(domain="frequency", deltaf=initilde.delta_f)):
    hp, hc = waveform

    hp.resize(len(initilde))
    snr = pycbc.filter.matched_filter(hp, initilde, psd=psd,
                                      low_frequency_cutoff=flow)

    # remove regions corrupted by filter wraparound (see https://pycbc.org/pycbc/latest/html/gw150914.html#calculate-the-signal-to-noise)
    snr = snr[len(snr) // 4: len(snr) * 3 // 4]

    plt.plot(snr.sample_times, abs(snr))
    plt.ylabel('signal-to-noise ratio')
    plt.xlabel('time (s)')
    plt.show()

    if i == 0:
        # just run for one loop
        break
