import numpy as np
import matplotlib.pyplot as plt

R_earth = 6371.0
roche_R = 2.9  # roche limit in Earth radii

plt.figure(figsize=(7.0, 3.0))
plt.axvspan(0, roche_R, alpha=0.25,color = 'lightcoral')      # inside Roche
plt.axvspan(roche_R, 10.0, alpha=0.25, color = 'lightblue')     # outside Roche
plt.axvline(roche_R, linestyle="--", linewidth=1.5)
plt.text((1 + roche_R) / 2, 0.65, "No clumping", ha="center", va="center")
plt.text((roche_R + 10.0) / 2, 0.65, "Accretion possible", ha="center", va="center")
plt.text(roche_R + 0.10, 0.12, f"Roche ≈ {roche_R:.1f} R⊕", ha="left", va="center")
plt.ylim(0, 1)
plt.yticks([])
plt.xlabel("Distance from Earth (R⊕)")
plt.xlim(1, 10.0)
plt.tight_layout()
plt.rcParams.update({"font.size": 34})
fig = plt.gcf()
ax = plt.gca()
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

plt.savefig("./roche_limit_moon_zone.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)

# plt.savefig("roche_limit_moon_zone.png", dpi=300, bbox_inches="tight")
plt.show()

