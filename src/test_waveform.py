"""
Show how to create a template bank equivalent to that in Fig. 7.5 (p. 78) of
arXiv:0908.2085.
"""

from ringdown import ringdown_waveform
from matplotlib import pyplot as pl
import pycbc.waveform


# add waveform to PyCBC
pycbc.waveform.add_custom_waveform(
    'ringdown', ringdown_waveform, 'time', force=True,
)


params = dict(
    q=5,
    freq=1234.0,
    iota=0.5,
    amplitude=1e-23,
)

# plot waveform
hp, hc = pycbc.waveform.get_td_waveform(approximant="ringdown",
                                        f_lower=20,
                                        delta_t=1.0/4096,
                                        **params)

fig, ax = pl.subplots(2, 1, figsize=(5, 9))
ax[0].plot(hp.sample_times, hp)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amplitude")

# plot frequency domain
hf = hp.to_frequencyseries()
ax[1].semilogx(hf.sample_frequencies, hf.real())
ax[1].set_xlabel('Frequency (Hz)')

fig.tight_layout()
pl.show()