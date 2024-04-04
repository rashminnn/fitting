import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image = cv.imread("Crop_field_cropped.jpg")
assert image is not None
img1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edges = cv.Canny(img1, 550, 690)

indices = np.where(edges != 0)
x = indices[1]
y = indices[0]

u11 = np.sum((x - np.mean(x))**2)
u12 = np.sum((x - np.mean(x))*(y - np.mean(y)))
u21 = u12
u22 = np.sum((y - np.mean(y))**2)
U = np.array([[u11, u12], [u21, u22]])

# total least squares solution
w, v = np.linalg.eig(U)
smallest_eigenvector = v[:, np.argmin(w)]
a = smallest_eigenvector[0]
b = smallest_eigenvector[1]
d = a*np.mean(x) + b*np.mean(y)
m = -a/b
c = d/b

y_new = m * x + c
rad = np.arctan(m)
deg = np.degrees(rad)

plt.scatter(x, y, 1)
plt.plot(x, y_new, color='r')
plt.title(f'Angle of the fitted line: {deg:.2f} degrees')
plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB))
# Enable this to see the line with cropped image
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.show()
