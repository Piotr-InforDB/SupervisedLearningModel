import numpy as np
import matplotlib.pyplot as plt

size = np.array([32, 64, 128, 256])
count = np.array([50, 100, 250, 500, 1000, 2000])
accuracy = np.array([
    [58, 62, 65, 67, 72, 74],
    [62, 64, 72, 74, 78, 81],
    [60, 74, 87, 85, np.nan, np.nan],
    [81, 82, np.nan, np.nan, np.nan, np.nan]
])

X, Y = np.meshgrid(count, size)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X_flat = X[~np.isnan(accuracy)]
Y_flat = Y[~np.isnan(accuracy)]
Z_flat = accuracy[~np.isnan(accuracy)]
ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='viridis', edgecolor='k')

for i in range(len(size)):
    valid_mask = ~np.isnan(accuracy[i])
    ax.plot(X[i, valid_mask], Y[i, valid_mask], accuracy[i, valid_mask], color='b', linewidth=0.5)

for j in range(len(count)):
    valid_mask = ~np.isnan(accuracy[:, j])
    ax.plot(X[valid_mask, j], Y[valid_mask, j], accuracy[valid_mask, j], color='b', linewidth=0.5)

ax.set_xlabel("Count")
ax.set_ylabel("Size")
ax.set_zlabel("Accuracy (%)")

plt.show()
