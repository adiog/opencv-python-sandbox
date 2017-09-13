import cv2
import math
import numpy as np
import sys

# INPUT PARAMETERS
target_x = 480
target_y = 680
side_cut_percent = 0.02
top_cut_percent = 0.018
bottom_cut_percent = 0.018
blur_before_segmentation = False
blur_radius = 3


def calculate_distance(a, b):
    return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]))


def find_minimal_bounding_rectangle(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def check_if_left_skewed(box):
    first_edge = calculate_distance(box[0], box[1])
    second_edge = calculate_distance(box[1], box[2])

    return first_edge / second_edge > 1





precise_framed_x = target_x / (1.0 - 2 * side_cut_percent)
side_cut_pixel = int(precise_framed_x * side_cut_percent)
framed_x = int(precise_framed_x)

precise_framed_y = target_y / (1.0 - top_cut_percent - bottom_cut_percent)
top_cut_pixel = int(precise_framed_y * top_cut_percent)
bottom_cut_pixel = int(precise_framed_y * bottom_cut_percent)
framed_y = int(precise_framed_y)

perpendicular_template_right_skewed_list = [(framed_x, framed_y), (0, framed_y), (0, 0), (framed_x, 0)]
perpendicular_template_left_skewed_list = perpendicular_template_right_skewed_list[-3:] + \
                                          perpendicular_template_right_skewed_list[:-3]

perpendicular_template_left_skewed_array = np.array(perpendicular_template_left_skewed_list)
perpendicular_template_right_skewed_array = np.array(perpendicular_template_right_skewed_list)

input_image = cv2.imread(sys.argv[1])
rows, cols, ch = input_image.shape
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
if blur_before_segmentation:
    gray_image_blur = cv2.blur(gray_image, (blur_radius, blur_radius))
else:
    gray_image_blur = gray_image
ret, threshold = cv2.threshold(gray_image_blur, 127, 255, cv2.THRESH_BINARY_INV)
im2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for index, contour in enumerate(contours):
    skewed_box = find_minimal_bounding_rectangle(contour)
    lefty = check_if_left_skewed(skewed_box)

    if lefty:
        perpendicular_template = perpendicular_template_left_skewed_array
    else:
        perpendicular_template = perpendicular_template_right_skewed_array

    # ---- Framing the homography matrix
    homography, status = cv2.findHomography(skewed_box, perpendicular_template)

    # ---- transforming the image bound in the rectangle to straighten
    framed_image = cv2.warpPerspective(input_image, homography, (framed_x, framed_y))

    crop_image = framed_image[top_cut_pixel:framed_y - bottom_cut_pixel, side_cut_pixel:framed_x - side_cut_pixel]

    output_filename = 'output' + str(index) + '.png'
    cv2.imwrite(output_filename, crop_image)
    print('Extracted ' + output_filename)

