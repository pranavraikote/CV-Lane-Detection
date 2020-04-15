# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:55:31 2020

@author: Pranav
"""
#Import necesaary functions and modules
from LaneDetector import filter_img_hsl, convert_grayscale, gaussian_blur, canny_edge_detector, region_of_interest, hough_transform, separate_lines, color_lanes, trace_both_lane_lines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2

#Defining parameter for Hough Transform
rho = 1
theta = (np.pi/180) * 1
threshold = 15
min_line_length = 20
max_line_gap = 10

#Function to define a pipeline to process all image operations 
def pipeline(image):
    #def pipeline(path)
    #t = time.time()
    #image = mpimg.imread(path)
    combined_hsl_img = filter_img_hsl(image)
    grayscale_img = convert_grayscale(combined_hsl_img)
    gaussian_smoothed_img = gaussian_blur(grayscale_img, kernel_size=5)
    canny_img = canny_edge_detector(gaussian_smoothed_img, 50, 150)
    segmented_img = region_of_interest(canny_img)
    hough_lines = hough_transform(segmented_img, rho, theta, threshold, min_line_length, max_line_gap)
    seperated_lanes = separate_lines(hough_lines, image)
    final_image = color_lanes(image, seperated_lanes[0], seperated_lanes[1])
    #final_image = trace_both_lane_lines(image, seperated_lanes[0], seperated_lanes[1])
    #print('Time taken is {}'.format(time.time()-t))
    #plt.imshow(final_image)
    return final_image
    
#path = 'dataset/video_challenge_0s.jpg'
#pipeline(path)

#Read the input Video Data 
cap = cv2.VideoCapture('video_dataset/solidWhiteRight.mp4')

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    #Display the processed frame
    a = pipeline(frame)
    #cv2.imshow('Frame',frame)
    cv2.imshow('Processing',a)

    #Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  else: 
    break

#Release the capturing object when done
cap.release()

#Destroy all windows
cv2.destroyAllWindows()