import matplotlib.pyplot as plt
import numpy as np
from colors import default_gamma, srgb_gamma, simple_gamma, no_dark_gamma, crazy_gamma, square_gamma, linear_gamma, inverse_gamma

x = np.linspace(0, 1.0, 25)
fig, ax = plt.subplots()
ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
ax.plot(x, default_gamma(x), linewidth=2.0, label="default_gamma")
ax.plot(x, srgb_gamma(x), linewidth=2.0, label="srgb_gamma")
ax.plot(x, simple_gamma(x), linewidth=2.0, label="simple_gamma")
ax.plot(x, no_dark_gamma(x), linewidth=2.0, label="no_dark_gamma")
ax.plot(x, crazy_gamma(x), linewidth=2.0, label="crazy_gamma")
ax.plot(x, square_gamma(x), linewidth=2.0, label="square_gamma")
ax.plot(x, linear_gamma(x), linewidth=2.0, label="linear_gamma")
ax.plot(x, inverse_gamma(x), linewidth=2.0, label="inverse_gamma")
ax.legend()
plt.show()
