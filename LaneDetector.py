# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:28:14 2020

@author: Pranav
"""
#Import necesaary functions and modules
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import stats
import cv2

#Generate the paths of the images
image_dir = 'dataset/'
paths = os.listdir(image_dir)
paths = list(map(lambda path: image_dir + path, paths))

#Open all images and store as np arrays
original_images = list(map(lambda img_name: mpimg.imread(img_name), paths))

#Function to show the image
def show_image(path):
    if type(path)==str:
        img = cv2.imread(path)
        plt.imshow(img)
    else:
        plt.imshow(path)

#Conversion to Hue Saturation Value
def convert_hsv(img):
    #img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

#Conversion to Hue Saturation Lightness
def convert_hsl(img):
    #img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

#Batch apply HSV & HSL on all images
hsv_images = list(map(lambda image: convert_hsv(image), original_images))
hsl_images = list(map(lambda image: convert_hsl(image), original_images))

#Function to define the White Colour Mask
def white_mask(image):
    l = np.array([0, 200, 0], dtype=np.uint8)
    h = np.array([180, 255, 255], dtype=np.uint8)
    
    white = cv2.inRange(image, l, h)
    return white

#Function to define the Yellow Colour Mask
def yellow_mask(image):
    l = np.array([15, 38, 115], dtype=np.uint8)
    h = np.array([35, 204, 255], dtype=np.uint8)
    
    white = cv2.inRange(image, l, h)
    return white

#Batch apply Mask on all images
hsl_white_images = list(map(lambda img: white_mask(img), hsl_images))
hsl_yellow_images = list(map(lambda img: yellow_mask(img), hsl_images))    

#Function to apply HSL with original images
def combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white):
    hsl_mask = cv2.bitwise_or(hsl_yellow, hsl_white)
    return cv2.bitwise_and(img, img, mask=hsl_mask)

#Function to Convert, Apply both White and Yellow masks on HSL
def filter_img_hsl(img):
    hsl_img = convert_hsl(img)
    hsl_yellow = yellow_mask(hsl_img)
    hsl_white = white_mask(hsl_img)
    return combine_hsl_isolated_with_original(img, hsl_yellow, hsl_white)

#Batch apply on all images
combined_hsl_images = hsl_images = list(map(lambda img: filter_img_hsl(img), original_images))

#Function to convert images to grayscale
def convert_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#Function to blur the images
def gaussian_blur(grayscale_img, kernel_size=3):
    return cv2.GaussianBlur(grayscale_img, (kernel_size, kernel_size), 0)

#Batch apply on all images
grayscale_images = list(map(lambda image: convert_grayscale(image), original_images)) 
blurred_images = list(map(lambda img: gaussian_blur(img, kernel_size=5), grayscale_images))

#Applying Canny Edge Detector
def canny_edge_detector(blurred_img, low_threshold, high_threshold):
    return cv2.Canny(blurred_img, low_threshold, high_threshold)

#Batch apply on all images
canny_images = list(map(lambda img: canny_edge_detector(img, 50, 150), blurred_images)) 

#Function to specify the Region of Interests and get the vertices
def get_vertices_for_img(img):
    imshape = img.shape
    height = imshape[0]
    width = imshape[1]

    vert = None
    
    if (width, height) == (960, 540):
        region_bottom_left = (130 ,imshape[0] - 1)
        region_top_left = (410, 330)
        region_top_right = (650, 350)
        region_bottom_right = (imshape[1] - 30,imshape[0] - 1)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)
    else:
        region_bottom_left = (200 , 680)
        region_top_left = (600, 450)
        region_top_right = (750, 450)
        region_bottom_right = (1100, 650)
        vert = np.array([[region_bottom_left , region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

    return vert

def region_of_interest(img):
    
    #Blank mask
    mask = np.zeros_like(img)   
        
    #Define a channel (1 or 3 channel)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    vert = get_vertices_for_img(img)    
        
    #Fill the RoI Polygon with vertices
    cv2.fillPoly(mask, vert, ignore_mask_color)
    
    #Return the masked images
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

segmented_images = list(map(lambda img: region_of_interest(img), canny_images))

#Applying Hough Transform
def hough_transform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

#Parameters for Hough Transform
rho = 1
theta = (np.pi/180) * 1
threshold = 15
min_line_length = 20
max_line_gap = 10

#Batch apply on all images
hough_lines_per_image = list(map(lambda img: hough_transform(img, rho, theta, threshold, min_line_length, max_line_gap), segmented_images))

#Draw lines based on the Hough Transform
def draw_lines(img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    
    return img_copy

#Batch apply on all images
img_with_lines = list(map(lambda img, lines: draw_lines(img, lines), original_images, hough_lines_per_image))

#Function to 
def separate_lines(lines, img):
    img_shape = img.shape
    
    middle_x = img_shape[1] / 2
    
    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1 
            if dx == 0:
                #Eliminate as it's undefined
                continue
            dy = y2 - y1
            
            #Eliminate if dy is constant
            if dy == 0:
                continue
            
            slope = dy / dx
            
            #Eliminating lines with small slope
            epsilon = 0.1
            if abs(slope) <= epsilon:
                continue
            
            if slope < 0 and x1 < middle_x and x2 < middle_x:
                #Lane should be within the LHS of RoI
                left_lane_lines.append([[x1, y1, x2, y2]])
            
            elif x1 >= middle_x and x2 >= middle_x:
                #Lane should be within the RHS of RoI
                right_lane_lines.append([[x1, y1, x2, y2]])
    
    return left_lane_lines, right_lane_lines

#Batch apply on all images
separated_lanes_per_image = list(map(lambda lines, img: separate_lines(lines, img), hough_lines_per_image, original_images))

#Function to apply color on lines
def color_lanes(img, left_lane_lines, right_lane_lines, left_lane_color=[255, 0, 0], right_lane_color=[0, 0, 255]):
    left_colored_img = draw_lines(img, left_lane_lines, color=left_lane_color, make_copy=True)
    final_img = draw_lines(left_colored_img, right_lane_lines, color=right_lane_color, make_copy=False)
    
    return final_img

#Batch apply on all images
img_different_lane_colors = list(map(lambda img, separated_lanes: color_lanes(img, separated_lanes[0], separated_lanes[1]), original_images, separated_lanes_per_image))

#Function to find lanes on that line plane
def find_lane_lines_formula(lines):
    xs = []
    ys = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    
    #Return Slope and Intercept
    return (slope, intercept)

#Function to tracing and extending the lines
def trace_lane_line(img, lines, top_y, make_copy=True):
    A, b = find_lane_lines_formula(lines)
    vert = get_vertices_for_img(img)

    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    x_to_bottom_y = (bottom_y - b) / A
    
    top_x_to_y = (top_y - b) / A 
    
    new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines, make_copy=make_copy)

#Function to tracing and extending both the lines
def trace_both_lane_lines(img, left_lane_lines, right_lane_lines):
    vert = get_vertices_for_img(img)
    region_top_left = vert[0][1]
    
    full_left_lane_img = trace_lane_line(img, left_lane_lines, region_top_left[1], make_copy=True)
    full_left_right_lanes_img = trace_lane_line(full_left_lane_img, right_lane_lines, region_top_left[1], make_copy=False)
    
    #Image 1 * Alpha + Image 2 * Beta + λ
    #Image 1 and Image 2 must be the same shape.
    img_with_lane_weight =  cv2.addWeighted(img, 0.7, full_left_right_lanes_img, 0.3, 0.0)
    
    return img_with_lane_weight

#Batch apply on all images
full_lane_drawn_images = list(map(lambda img, separated_lanes : trace_both_lane_lines(img, separated_lanes[0], separated_lanes[1]), original_images, separated_lanes_per_image))

#Function to define a pipeline to process all image operations 
def pipeline(image):
    combined_hsl_img = filter_img_hsl(image)
    grayscale_img = convert_grayscale(combined_hsl_img)
    gaussian_smoothed_img = gaussian_blur(grayscale_img, kernel_size=5)
    canny_img = canny_edge_detector(gaussian_smoothed_img, 50, 150)
    segmented_img = region_of_interest(canny_img)
    hough_lines = hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)
    seperated_lanes = separate_lines(hough_lines, image)
    #final_image = color_lanes(image, seperated_lanes[0], seperated_lanes[1])
    #plt.imshow(final_image)
    final_image = trace_both_lane_lines(image, seperated_lanes[0], seperated_lanes[1])
    plt.imshow(final_image)