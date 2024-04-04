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

# Least square fitting
u11 = np.sum((x - np.mean(x))**2)
u12 = np.sum((x - np.mean(x))*(y - np.mean(y)))

# finding the best slope and the intersection for the least squared fitting
m = u12/u11
c = np.mean(y)-m*(np.mean(x))
y_new = m*x+c

rad = np.arctan(m)
deg = np.degrees(rad)

plt.gca().invert_yaxis()  # inverting y axis because thats how image pixels measures
plt.scatter(x, y, 1)
plt.plot(x, y_new, color='red')
plt.title(f'Angle of the fitted line: {deg:.2f} degrees')
plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB))

# {to get the full image with the line enable this part}
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

plt.show()
