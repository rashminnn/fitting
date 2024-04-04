import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from sklearn import linear_model

image = cv.imread("Crop_field_cropped.jpg")
assert image is not None
img1 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edges = cv.Canny(img1, 550, 690)

indices = np.where(edges != 0)
# converting the x values to a vector to give it as a ransac input
X = indices[1].reshape(-1, 1)
y = indices[0]

# Fitting line
lr = linear_model.LinearRegression()
lr.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)

# calculating the angle of the line
slope = ransac.estimator_.coef_[0]
deg = np.degrees(np.arctan(slope))

plt.scatter(
    X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
)
plt.plot(
    line_X,
    line_y_ransac,
    color="red",
    linewidth=2,
    label="RANSAC regressor",
)
plt.gca().invert_yaxis()
plt.legend(loc="lower right")
plt.imshow(cv.cvtColor(edges, cv.COLOR_BGR2RGB))
# Enable this to see the line with cropped image
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title(f'Angle of the RANSAC line is {deg:.2f}')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
