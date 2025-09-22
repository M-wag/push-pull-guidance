import numpy as np
import matplotlib.pyplot as plt 
from mylib.diffusion import time_steps_edm
t = time_steps_edm(num_steps=32, sigma_min=0.002, sigma_max=80, rho=7).tolist()

fig, ax = plt.subplots()
intervals = [
    (46.19, 80, "gray", "Neglible Effect"),
    (16.59, 46.19, "yellow", "Start Transition"),
    (6.46, 16.59, "green", "Viable Results"),
    (0.0, 6.46, "red", "Noisy Template")
]

for left, right, color, label in intervals:
    ax.axhspan(left, right, color=color, alpha=0.2)
    # put label in the middle of the span
    mid = (left + right) / 2
    ax.text(2.5, mid, label, ha="center", va="center", fontsize=10, alpha=0.8)
ax.scatter(range(0, len(t)), np.array(t)[::-1], alpha=0.8)
ax.set_xlabel("Index")
ax.set_ylabel("Time")
ax.set_yscale('log')
ax.grid(True)

x = []
y = []

ax.legend()



plt.title("Time steps for EDM sampler (steps=32)")
plt.savefig('temp.png')


