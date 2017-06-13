import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

input_path = "autoencoded_lfw_samples"

ax1 = plt.subplot(121, aspect='equal', adjustable='box-forced')
ax1.set_title("maaGMA Style Samples")
gaussians = pickle.load(open(os.path.join(input_path, "gaussians.pkl"), "rb"))
plt.plot(gaussians[:, 1], gaussians[:, 2], color='green', marker='o', markersize=1, linewidth=0)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")


ax2 = plt.subplot(122, aspect='equal', adjustable='box-forced', sharex=ax1, sharey=ax1)
ax2.set_title("Gaussian Samples")
gaussians = np.random.normal(size=(1000, 10))
plt.plot(gaussians[:, 1], gaussians[:, 2], color='black', marker='o', markersize=1, linewidth=0)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.tight_layout()

plt.savefig(os.path.join(input_path, "gaussians.png"))
