import numpy as np
import matplotlib.pyplot as plt




def get_slopes(edges):
    # get the vectors    
    x1 = edges[:, :, 0]
    y1 = edges[:, :, 1]
    x2 = edges[:, :, 2]
    y2 = edges[:, :, 3]
    v = np.hstack([x2-x1, y2-y1])
    unit_vectors = (v.T / np.linalg.norm(v, axis=1)).T # had to rework this for vector math
    return unit_vectors[:,1] / unit_vectors[:,0]

# The array of points in x1, y1, x2, y2 format
points = np.array([[[553,  3, 578,  61]],
                   [[607, 131, 630, 190]],
                   [[631, 194, 653, 254]],
                   [[854, 449, 872, 485]],
                   [[898, 610, 914, 632]],
                   [[553,  3, 653+200, 254+502]],])
print(get_slopes(points))
# Reshape the array to remove the extra dimension
points = points.reshape(-1, 4)

# Plotting each line segment
plt.figure(figsize=(8, 6))




h, w = points.shape
x1 = points[:, 0]
y1 = points[:, 1]
x2 = points[:, 2]
y2 = points[:, 3]
xs = np.vstack([x1, x2]).flatten()
print(xs)
ys = np.vstack([y1, y2]).flatten()
coefficients = np.polyfit(xs, ys, deg=1) # highest power first
y = np.polyval(coefficients, xs)



for point in points:
    x1, y1, x2, y2 = point
    plt.plot([x1, x2], [y1, y2], marker='o')
plt.scatter(xs, y, marker='x')

# Customize the plot
plt.title('Line Segments')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.gca().invert_yaxis()  # Invert the Y-axis if needed for image-like plotting
plt.show()