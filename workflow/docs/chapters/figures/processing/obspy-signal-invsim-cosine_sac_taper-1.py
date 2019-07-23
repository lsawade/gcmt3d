"""
Cosine taper plot

Original plot from obspy website.

Modified by Lucas Sawade, June 2019

"""

import matplotlib.pylab as plt
import numpy as np
from obspy.signal.invsim import cosine_sac_taper

plt.figure(figsize=(10, 3))

freqs = np.logspace(-2.01, 0, 2000)

plt.vlines([0.015, 0.03, 0.2, 0.4], -0.1, 1.3, color="#89160F")
plt.semilogx(freqs, cosine_sac_taper(freqs, (0.015, 0.03, 0.2, 0.4)),
             lw=2, color="#4C72B0")

props = {
    "bbox": dict(facecolor='white', edgecolor="0.5",
         boxstyle="square,pad=0.2"),
    "va": "top", "ha": "center", "color": "#89160F",
    "size": "large"}

plt.text(0.015, 1.25, "f1", **props)
plt.text(0.03, 1.25, "f2", **props)
plt.text(0.2, 1.25, "f3", **props)
plt.text(0.4, 1.25, "f4", **props)

plt.xlim(freqs[0], freqs[-1])
plt.ylim(-0.1, 1.3)
plt.ylabel("Taper Amplitude")
plt.xlabel("Frequency [Hz]")
plt.grid()
plt.tight_layout()
plt.savefig(fname="cosine_taper.png", dpi=150, format="png")
plt.show()