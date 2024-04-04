# test
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# total least squares estimation
image = cv.imread("Crop_field_cropped.jpg")
img1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edges = cv.Canny(img1, 550, 690)

indices = np.where(edges != 0)
x = indices[1]
y = indices[0]


X = np.stack((x, y), axis=1)
# sort eigen values largest to smallest
l, v = np.linalg.eig(X.T@X)
idx = l.argsort()[::-1]
u11 = v[:, idx[-1]]                          # smallest eigen value
u12 = np.array((u11[1], -u11[0]))            # rotate normal vector by 90

plt.scatter(x, y, 1)
plt.plot([np.min(X[:, 0]), np.max(X[:, 0])],
         [np.min(X[:, 0]) * u12[1] / u12[0],
          np.max(X[:, 0]) * u12[1] / u12[0]],
         color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Total Least Squares Line')
plt.show()
