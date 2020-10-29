"""
Show how to create a template bank equivalent to that in Fig. 7.5 (p. 78) of
arXiv:0908.2085.
"""

from ringdown import RingdownTemplateBank
from matplotlib import pyplot as pl


flow = 50  # low frequency (Hz)
fhigh = 2000  # high frequency (Hz)
qrange = [2, 20]  # Quality factor ranges
mm = 0.03  # maximum mismatch

tb = RingdownTemplateBank(flow, fhigh, qrange=qrange, mm=mm)

print("Number of templates is {}".format(len(tb)))

fig, ax = pl.subplots()
ax.semilogx(tb.bank_freqs, tb.bank_qs, '.', color="b", ls="None")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Q")
ax.grid(True, which="both", linestyle="dotted")
pl.show()
