# This class was adapted from Martin Evans's (https://stackoverflow.com/users/4985733/martin-evans)
# Stack Overflow answer to Max Gordon's question (https://stackoverflow.com/users/409508/max-gordon)
# The original answer can be found here: https://stackoverflow.com/questions/32435488/align-x-ray-images-find-rotation-rotate-and-crop
import cv2
import numpy as np
from PIL import Image
import math


class AdjustImage(object):
    THRESHOLD = 240

    def subimage(self, image, center, theta, width, height):
        if 45 < theta <= 90:
            theta = theta - 90
            width, height = height, width

        theta *= math.pi / 180  # convert to rad
        v_x = (math.cos(theta), math.sin(theta))
        v_y = (-math.sin(theta), math.cos(theta))
        s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
        s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)
        mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])
        return cv2.warpAffine(image, mapping, (width, height), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    def __call__(self, image_source):

        # Save the original image for error handling later
        # Can see if it came in corrupted or something down the line affected it
        image_source_original = image_source
        image_source = np.asarray(image_source)
        init_crop = 5
        dimensions = image_source.ndim
        h, w = image_source.shape[:2]

        # crop out and add back border for additional definition of x-ray vs image
        image_source = image_source[init_crop:init_crop +
                                    (h-init_crop*2), init_crop:init_crop+(w-init_crop*2)]
        # Add back a white border
        image_source = cv2.copyMakeBorder(
            image_source, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Check if image is already in greyscale or not, if not convert it
        if dimensions == 2:
            image_grey = image_source.copy()
        else:
            image_grey = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
        # Create the thresh of the the greyscale image - using the threshold of TOZERO_INV (a gradient that cuts to 0)
        _, image_thresh = cv2.threshold(
            image_grey, self.THRESHOLD, 255, cv2.THRESH_TOZERO_INV)
        image_thresh2 = image_thresh.copy()

        # Get the canny edge detection
        image_thresh2 = cv2.Canny(image_thresh2, 100, 100, apertureSize=3)

        # Get the points from the canny edge detection - sometimes this may error
        # Sometimes the shape found isn't a rectangle, as the corners of the x-ray are obscured/cut off from
        # augmentation. In that case, the original image is returned. The neural network can handle rotation/shear
        # if it has to
        points = cv2.findNonZero(image_thresh2)

        # Get the base rectangle of the x-ray
        centre, dimensions, theta = cv2.minAreaRect(points)
        rect = cv2.minAreaRect(points)

        width = int(dimensions[0])
        height = int(dimensions[1])

        box = cv2.boxPoints(rect)
        box = np.intp(box)

        temp = image_source.copy()
        cv2.drawContours(temp, [box], 0, (255, 0, 0), 2)

        M = cv2.moments(box)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # It assumes the image is already incorrectly rotated - so angle of image + 90 to fix it.
        image_patch = self.subimage(
            image_source, (cx, cy), (theta+90), height, width)

        # add back a small border
        image_patch = cv2.copyMakeBorder(
            image_patch, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Now find the thresh of the original sub image + do edge detection
        _, image_thresh = cv2.threshold(image_patch, self.THRESHOLD, 255, 1)
        image_thresh = cv2.Canny(image_thresh, 100, 100, 3)
        points = cv2.findNonZero(image_thresh)
        hull = cv2.convexHull(points)

        # Find the smallest set of points that contains the image
        for epsilon in range(3, 50):
            hull_simple = cv2.approxPolyDP(hull, epsilon, 1)

            if len(hull_simple) == 4:
                break

        hull = hull_simple

        # Find closest fitting image size and warp/crop to fit

        x, y, w, h = cv2.boundingRect(hull)
        target_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

        # Sort hull into top left,top right,bottom right,bottom left order.

        source_corners = hull.reshape(-1, 2).astype('float32')
        min_dist = 100000
        index = 0

        for n in range(len(source_corners)):
            x, y = source_corners[n]
            dist = math.hypot(x, y)

            if dist < min_dist:
                index = n
                min_dist = dist

        # Rotate the array so top left is first

        source_corners = np.roll(source_corners, -(2*index))

        try:
            # return the adjusted image
            transform = cv2.getPerspectiveTransform(
                np.float32(source_corners), np.float32(target_corners))
            var = Image.fromarray(cv2.warpPerspective(
                image_patch, transform, (w, h))).convert('RGB')
            return var
        except:
            # We want to return the original image - occasionally (mainly with augmented images)
            # the generated thresh would have more than 4 points. In that case, it's fine for
            # an image not to be adjusted.
            return image_source_original

    def __repr__(self):
        return self.__class__.__name__+'()'
