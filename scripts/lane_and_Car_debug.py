import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
import edge_detection as edge # Handles the detection of lane lines
import matplotlib.pyplot as plt # Used for plotting and error checking
from masks import feature_mask
from PIL import Image
from ultralytics import YOLO
import os
import pandas as pd
import re
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as ExcelImage
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from pathlib import Path
import json
from reportlab.lib.units import cm
 

class Lane:
  """
  Represents a lane on a road.
  """
  def __init__(self, orig_frame,img_path, road_shape,ego_id,video_folder,output_folder):
    """
      Default constructor
         
    :param orig_frame: Original camera image (i.e. frame)
    :param ego_id: ID of the ego vehicle
    """
    self.video_folder = video_folder
    self.img = Image.open(img_path)
    self.orig_frame = orig_frame
    self.ego_id = ego_id
    self.output_folder = output_folder

    # This will hold an image with the lane lines       
    self.lane_line_markings = None
 
    # This will hold the image after perspective transformation
    self.warped_frame = None
    self.transformation_matrix = None
    self.inv_transformation_matrix = None
    # (Width, Height) of the original video frame (or image)
    self.orig_image_size = self.orig_frame.shape[::-1][1:]
 
    width = self.orig_image_size[0]
    height = self.orig_image_size[1]
    self.width = width
    self.height = height

    self.straight = True if road_shape =='straight' else False
    self.slight_curve_right = True if road_shape =='slight_curve_right' else False
    self.slight_curve_left = True if road_shape =='slight_curve_left' else False
    self.curve_right = True if road_shape =='right_curve' else False
    self.curve_left = True if road_shape =='left_curve' else False

    ROI_FILE = self.output_folder / f"selected_rois_{self.video_folder}.json"

    if not ROI_FILE.exists():
        raise FileNotFoundError("❌ selected_rois.json introuvable. Veuillez définir les ROI.")
    
    with open(ROI_FILE, "r", encoding="utf-8") as f:
        roi_config = json.load(f)


    # Four corners of the trapezoid-shaped region of interest
    # You need to find these corners manually.
    if not ROI_FILE.exists():
      if self.straight :
        self.roi_points = np.float32([
          (600,410), # Top-left corner
          (300, 598), # Bottom-left corner            
          (1030,580), # Bottom-right corner
          (715,410) # Top-right corner
        ])

      if self.slight_curve_right :
        self.roi_points = np.float32([
          (590,410), # Top-left corner
          (300, 598), # Bottom-left corner            
          (1030,580), # Bottom-right corner
          (730,410) # Top-right corner
        ])

      if self.slight_curve_left :
        self.roi_points = np.float32([
          (550,410), # Top-left corner
          (300, 598), # Bottom-left corner            
          (1030,580), # Bottom-right corner
          (700,410) # Top-right corner
        ])

      if self.curve_left :
        self.roi_points = np.float32([
          (500,423), # Top-left corner
          (300, 603), # Bottom-left corner            
          (1100,580), # Bottom-right corner
          (680,414) # Top-right corner
        ])
      if self.curve_right :
        self.roi_points = np.float32([
          (580,423), # Top-left corner
          (200, 603), # Bottom-left corner            
          (1010,580), # Bottom-right corner
          (750,407) # Top-right corner
        ])
    else :
      for ego_id, shapes in roi_config.items():
        if ego_id == self.ego_id :
          if self.straight :
            self.roi_points =np.array(roi_config[self.ego_id]["straight"]["points"], dtype=np.float32)
          if self.slight_curve_right :
            self.roi_points = np.array(roi_config[self.ego_id]["slight_curve_right"]["points"], dtype=np.float32)
          if self.slight_curve_left :
            self.roi_points = np.array(roi_config[self.ego_id]["slight_curve_left"]["points"], dtype=np.float32)
          if self.curve_left :
            self.roi_points = np.float32([
                (500,423), # Top-left corner
                (300, 603), # Bottom-left corner            
                (1100,580), # Bottom-right corner
                (680,414) # Top-right corner
              ])
          if self.curve_right :
            self.roi_points = np.float32([
                [594, 400],
                [0, 465],
                [1277, 550],
                [733, 397]
            ])
        
    # The desired corner locations  of the region of interest
    # after we perform perspective transformation.
    # Assume image width of 600, padding == 150.
    self.padding = int(0.25 * width) # padding from side of the image in pixels
    self.desired_roi_points = np.float32([
      [self.padding, 0], # Top-left corner
      [self.padding, self.orig_image_size[1]], # Bottom-left corner         
      [self.orig_image_size[
        0]-self.padding, self.orig_image_size[1]], # Bottom-right corner
      [self.orig_image_size[0]-self.padding, 0] # Top-right corner
    ]) 
      
    # Histogram that shows the white pixel peaks for lane line detection
    self.histogram = None
         
    # Sliding window parameters
    self.no_of_windows = 10
    self.margin = int((1/12) * width)  # Window width is +/- margin
    self.minpix = int((1/24) * width)  # Min no. of pixels to recenter window
         
    # Be70t fit polynomial lines for left line and right line of the lane
    self.left_fit = None
    self.right_fit = None
    self.left_lane_inds = None
    self.right_lane_inds = None
    self.ploty = None
    self.left_fitx = None
    self.right_fitx = None
    self.leftx = None
    self.rightx = None
    self.lefty = None
    self.righty = None

    self.degree_one = False

    self.thresh_big = 190
    self.thresh_small = 150
    self.detected_lanes = False
    self.numberof_lanesdetected = 0
    self.passed_validation_test = False

    self.pts_center_orig = None
    
    self.distance_lat_excel = None
    self.distance_long_excel = None
    self.car_candidates = []

    self.matching_detection = None
    # Pixel parameters for x and y dimensions
    self.YM_PER_PIX = 45 / 170 # meters per pixel in y dimension
    self.XM_PER_PIX = 3 / 700 # meters per pixel in x dimension
         
    # Radii of curvature and offset
    self.left_curvem = None
    self.right_curvem = None
    self.center_offset = None
    
    self.road_shape = road_shape
    self.numberof_lanesdetected = 0

  def crop_to_roi(self, frame=None,plot=False):
    if frame is None :
      frame = self.lane_line_markings
    # Create a mask with the same dimensions as the input frame, initialized to zeros (black)
    mask = np.zeros_like(frame)

    # Fill the polygon defined by roi_points with white color (255)
    cv2.fillPoly(mask, [self.roi_points.astype(np.int32)], 255)

    # Perform a bitwise AND operation between the input frame and the mask
    # This will keep only the region of interest and set everything else to black
    cropped_frame = cv2.bitwise_and(frame,mask)
    
    if plot :
      cv2.imshow('crop', cropped_frame)
      cv2.waitKey(0)

    return cropped_frame

  def calculate_car_position(self, print_to_terminal=False):
    """
    Calculate the position of the car relative to the center
         
    :param: print_to_terminal Display data to console if True       
    :return: Offset from the center of the lane
    """
    if self.left_fit is not None and self.right_fit is not None:
      # Assume the camera is centered in the image.
      # Get position of car in centimeters
      car_location = self.orig_frame.shape[1] / 2
  
      # Fine the x coordinate of the lane line bottom
      height = self.orig_frame.shape[0]
      if self.degree_one :
        bottom_left = self.left_fit[0]*height + self.left_fit[1]
        bottom_right = self.right_fit[0]*height + self.right_fit[1]
        
      else :
        bottom_left = self.left_fit[0]*height**2 + self.left_fit[
          1]*height + self.left_fit[2]
        bottom_right = self.right_fit[0]*height**2 + self.right_fit[
          1]*height + self.right_fit[2]
  
      center_lane = (bottom_right - bottom_left)/2 + bottom_left 
      center_offset = (np.abs(car_location) - np.abs(
        center_lane)) * self.XM_PER_PIX * 100
  
      if print_to_terminal == True:
        print(str(center_offset) + 'cm')
              
      self.center_offset = center_offset
        
      return center_offset
    else:
      pass
 
  def calculate_curvature(self, print_to_terminal=False):
      """
      Calculate the road curvature in meters.
  
      :param: print_to_terminal Display data to console if True
      :return: Radii of curvature
      """
      # Set the y-value where we want to calculate the road curvature.
      # Select the maximum y-value, which is the bottom of the frame.
      if self.lefty is not None and self.righty is not None:  
      
        y_eval = np.max(self.ploty)    
        # Fit polynomial curves to the real world environment
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (
          self.XM_PER_PIX), 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (
          self.XM_PER_PIX), 2)
                
        # Calculate the radii of curvature
        left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*self.YM_PER_PIX + left_fit_cr[
                        1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvem = ((1 + (2*right_fit_cr[
                        0]*y_eval*self.YM_PER_PIX + right_fit_cr[
                        1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        # Display on terminal window
        if print_to_terminal == True:
          print(left_curvem, 'm', right_curvem, 'm')
                
        self.left_curvem = left_curvem
        self.right_curvem = right_curvem
    
        return left_curvem, right_curvem
      else:
        pass        
         
  def calculate_histogram(self,frame=None,plot=True):
    """
    Calculate the image histogram to find peaks in white pixel count
         
    :param frame: The warped image
    :param plot: Create a plot if True
    """
    if frame is None:
      frame = self.warped_frame
             
    # Generate the histogram
    self.histogram = np.sum(frame[int(
              frame.shape[0]/2):,:], axis=0)
 
    if plot == True:
         
      # Draw both the image and the histogram
      figure, (ax1, ax2) = plt.subplots(2,1) # 2 row, 1 columns
      figure.set_size_inches(10, 5)
      ax1.imshow(frame, cmap='gray')
      ax1.set_title("Warped Binary Frame")
      ax2.plot(self.histogram)
      ax2.set_title("Histogram Peaks")
      plt.show()
             
    return self.histogram
 
  def display_curvature_offset(self, frame=None, plot=False):
      """
      Display curvature and offset statistics on the image
          
      :param: plot Display the plot if True
      :return: Image with lane lines and curvature
      """
      #sprint(self.left_curvem, 'm', self.right_curvem, 'm', self.center_offset, 'cm')
      if self.left_curvem is not None and self.right_curvem is not None :
        image_copy = None
        if frame is None:
          image_copy = self.orig_frame.copy()
        else:
          image_copy = frame
    
        cv2.putText(image_copy,'Curve Radius: '+str((
          self.left_curvem+self.right_curvem)/2)[:7]+' m', (int((
          5/600)*self.width), int((
          20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
          0.5/600)*self.width)),(
          255,255,255),2,cv2.LINE_AA)
        cv2.putText(image_copy,'Center Offset: '+str(
          self.center_offset)[:7]+' cm', (int((
          5/600)*self.width), int((
          40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
          0.5/600)*self.width)),(
          255,255,255),2,cv2.LINE_AA)
                
        if plot==True:       
          cv2.imshow("Image with Curvature and Offset", image_copy)
    
        return image_copy
      else:
        pass
     
  def get_lane_line_previous_window(self, left_fit, right_fit, plot=True):
    # margin is a sliding window parameter
    margin = self.margin
    
    img_w = self.warped_frame.shape[1]
    center_x = img_w / 2

    # Find the x and y coordinates of all the nonzero 
    # (i.e. white) pixels in the frame.         
    nonzero = self.warped_frame.nonzero()  
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    if left_fit is not None and right_fit is not None:     
      # Store left and right lane pixel indices
      left_lane_inds = ((nonzerox > (left_fit[0]*(
        nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0]*(
        nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
      right_lane_inds = ((nonzerox > (right_fit[0]*(
        nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0]*(
        nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))           
      self.left_lane_inds = left_lane_inds
      self.right_lane_inds = right_lane_inds
  
      # Get the left and right lane line pixel locations  
      leftx = nonzerox[left_lane_inds]
      lefty = nonzeroy[left_lane_inds] 
      rightx = nonzerox[right_lane_inds]
      righty = nonzeroy[right_lane_inds]  
  
      self.leftx = leftx
      self.rightx = rightx
      self.lefty = lefty
      self.righty = righty        
      
      # Fit a second order polynomial curve to each lane line
      left_fit = np.polyfit(lefty, leftx, 2)
      right_fit = np.polyfit(righty, rightx, 2)       
      ploty = np.linspace(0, self.warped_frame.shape[0], self.warped_frame.shape[0]) 
      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
      self.ploty = ploty
      self.left_fitx = left_fitx
      self.right_fitx = right_fitx
      #print('left',left_fit,'right',right_fit)

      self.left_fit = left_fit
      self.right_fit = right_fit
            
    elif left_fit is not None and right_fit is None:
      lane_width_in_pixels = 500
       # Store left and right lane pixel indices
      left_lane_inds = ((nonzerox > (left_fit[0]*(
        nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0]*(
        nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))          
      self.left_lane_inds = left_lane_inds
      self.right_lane_inds = None
  
      # Get the left and right lane line pixel locations  
      leftx = nonzerox[left_lane_inds]
      lefty = nonzeroy[left_lane_inds]
      if self.straight or self.slight_curve_left or self.slight_curve_right: 
        rightx = 2 * center_x - leftx
        rightx = np.clip(rightx, 0, img_w-1)
      else :
        rightx = leftx + lane_width_in_pixels
      righty = lefty.copy()
  
      self.leftx = leftx
      self.rightx = rightx
      self.lefty = lefty
      self.righty = righty        
      
      # Fit a second order polynomial curve to each lane line
      left_fit = np.polyfit(lefty, leftx, 2)
      right_fit = np.polyfit(righty, rightx, 2)       
      ploty = np.linspace(0, self.warped_frame.shape[0], self.warped_frame.shape[0]) 
      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
      self.ploty = ploty
      self.left_fitx = left_fitx
      self.right_fitx = right_fitx
      #print('left',left_fit,'right',right_fit)

    elif left_fit is None and right_fit is not None:
      lane_width_in_pixels = 500
       # Store left and right lane pixel indices
      right_lane_inds = ((nonzerox > (right_fit[0]*(
        nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0]*(
        nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))           
      self.left_lane_inds = None
      self.right_lane_inds = right_lane_inds       

  
      rightx = nonzerox[right_lane_inds]
      righty = nonzeroy[right_lane_inds]
      if self.straight or self.slight_curve_left or self.slight_curve_right:
        leftx = 2 * center_x - rightx
        leftx = np.clip(leftx, 0, img_w-1)
      else :  
        leftx = rightx - lane_width_in_pixels
      lefty = righty.copy()
  
      self.leftx = leftx
      self.rightx = rightx
      self.lefty = lefty
      self.righty = righty        
      
      # Fit a second order polynomial curve to each lane line
      left_fit = np.polyfit(lefty, leftx, 2)
      right_fit = np.polyfit(righty, rightx, 2)       
      ploty = np.linspace(0, self.warped_frame.shape[0], self.warped_frame.shape[0]) 
      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
      self.ploty = ploty
      self.left_fitx = left_fitx
      self.right_fitx = right_fitx
      #print('left',left_fit,'right',right_fit)
         
    if plot==True:
         
      # Generate images to draw on
      out_img = np.dstack((self.warped_frame, self.warped_frame, (
                           self.warped_frame)))*255
      window_img = np.zeros_like(out_img)
             
      # Add color to the left and right line pixels
      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                                                                     0, 0, 255]
      # Create a polygon to show the search window area, and recast 
      # the x and y points into a usable format for cv2.fillPoly()
      margin = self.margin
      left_line_window1 = np.array([np.transpose(np.vstack([
                                    left_fitx-margin, ploty]))])
      left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                    left_fitx+margin, ploty])))])
      left_line_pts = np.hstack((left_line_window1, left_line_window2))
      right_line_window1 = np.array([np.transpose(np.vstack([
                                     right_fitx-margin, ploty]))])
      right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                     right_fitx+margin, ploty])))])
      right_line_pts = np.hstack((right_line_window1, right_line_window2))
             
      # Draw the lane onto the warped blank image
      cv2.fillPoly(window_img, np.int32([left_line_pts]), (0,255, 0))
      cv2.fillPoly(window_img, np.int32([right_line_pts]), (0,255, 0))
      result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
       
      # Plot the figures 
      figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(self.warped_frame, cmap='gray')
      ax3.imshow(result)
      ax3.plot(left_fitx, ploty, color='yellow')
      ax3.plot(right_fitx, ploty, color='yellow')
      ax1.set_title("Original Frame")  
      ax2.set_title("Warped Frame")
      ax3.set_title("Warped Frame With Search Window")
      plt.show()
    return left_fit, right_fit,left_fitx, right_fitx, ploty        
  
  def get_lane_line_indices_sliding_windows(self, plot=True):
    margin = self.margin
 
    frame_sliding_window = self.warped_frame.copy()
 
    # Set the height of the sliding windows
    window_height = int(self.warped_frame.shape[0]/self.no_of_windows)       
 
    # Find the x and y coordinates of all the nonzero 
    # (i.e. white) pixels in the frame. 
    nonzero = self.warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1]) 
         
    # Store the pixel indices for the left and right lane lines
    left_lane_inds = []
    right_lane_inds = []
         
    # Current positions for pixel indices for each window,
    # which we will continue to update
    leftx_base, rightx_base = self.histogram_peak()
    leftx_current = leftx_base
    rightx_current = rightx_base
 
    # Go through one window at a time
    no_of_windows = self.no_of_windows
         
    for window in range(no_of_windows):
       
      # Identify window boundaries in x and y (and right and left)
      win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
      win_y_high = self.warped_frame.shape[0] - window * window_height
      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(
        win_xleft_high,win_y_high), (255,255,255), 2)
      cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(
        win_xright_high,win_y_high), (255,255,255), 2)
 
      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (
                           nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (
                            nonzerox < win_xright_high)).nonzero()[0]
                                                         
      # Append these indices to the lists
      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)
         
      # If you found > minpix pixels, recenter next window on mean position
      minpix = self.minpix
      if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:        
        rightx_current = int(np.mean(nonzerox[good_right_inds]))
                     
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
 
    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds] 
    righty = nonzeroy[right_lane_inds]
 

    # LEFT LANE
    if len(leftx) == 0 or len(lefty) == 0:
        print("⚠️  Pas de voie gauche détectée")
        left_fit = None
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        self.numberof_lanesdetected += 1

    # RIGHT LANE
    if len(rightx) == 0 or len(righty) == 0:
        print("⚠️  Pas de voie droite détectée")
        right_fit = None
    else:
        right_fit = np.polyfit(righty, rightx, 2)
        self.numberof_lanesdetected += 1
    
    if left_fit is not None or right_fit is not None:
      self.detected_lanes = True
    elif left_fit is None and right_fit is None:
      self.detected_lanes = False


 
    if plot==True:
      # Create the x and y values to plot on the image  
      ploty = np.linspace(0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

      # Generate an image to visualize the result
      out_img = np.dstack((
        frame_sliding_window, frame_sliding_window, (
        frame_sliding_window))) * 255
            
      # Add color to the left line pixels and right line pixels
      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
        0, 0, 255]
                
      # Plot the figure with the sliding windows
      figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(frame_sliding_window, cmap='gray')
      ax3.imshow(out_img)
      ax3.plot(left_fitx, ploty, color='yellow')
      ax3.plot(right_fitx, ploty, color='yellow')
      ax1.set_title("Original Frame")  
      ax2.set_title("Warped Frame with Sliding Windows")
      ax3.set_title("Detected Lane Lines with Sliding Windows")
      plt.show()        

    self.left_fit = left_fit 
    self.right_fit = right_fit     
    return self.left_fit, self.right_fit
 
  def get_line_markings(self, frame=None,plot=True):
    """
    Isolates lane lines.
   
      :param frame: The camera frame that contains the lanes we want to detect
    :return: Binary (i.e. black and white) image containing the lane lines.
    """
    if frame is None:
      frame = self.orig_frame
             
    # Convert the video frame from BGR (blue, green, red) 
    # color space to HLS (hue, saturation, lightness).
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
 
    ################### Isolate possible lane line edges ######################
         
    # Perform Sobel edge detection on the L (lightness) channel of 
    # the image to detect sharp discontinuities in the pixel intensities 
    # along the x and y axis of the video frame.             
    # sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
    # Relatively light pixels get made white. Dark pixels get made black.
    _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
    sxbinary = edge.blur_gaussian(sxbinary, ksize=3) # Reduce noise
         
    # 1s will be in the cells with the highest Sobel derivative values
    # (i.e. strongest lane line edges)
    sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(100, 255))
 
    ######################## Isolate possible lane lines ######################
   
    # Perform binary thresholding on the S (saturation) channel 
    # of the video frame. A high saturation value means the hue color is pure.
    # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
    # and have high saturation channel values.
    # s_binary is matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the purest hue colors (e.g. >80...play with
    # this value for best results).
    s_channel = hls[:, :, 2] # use only the saturation channel data
    _, s_binary = edge.threshold(s_channel, (18, 220))
     
    # Perform binary thresholding on the R (red) channel of the 
    # original BGR video frame. 
    # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
    # White in the regions with the richest red channel values (e.g. >120).
    # Remember, pure white is bgr(255, 255, 255).
    # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
    _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))
 
    # Lane lines should be pure in color and have high red channel values 
    # Bitwise AND operation to reduce noise and black-out any pixels that
    # don't appear to be nice, pure, solid colors (like white or yellow lane 
    # lines.)       
    rs_binary = cv2.bitwise_and(s_binary, r_thresh)
 
    ### Combine the possible lane lines with the possible lane line edges ##### 
    # If you show rs_binary visually, you'll see that it is not that different 
    # from this return value. The edges of lane lines are thin lines of pixels.
    self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(
                              np.uint8))
    if plot==True:
      cv2.imshow("Lane Line Markings", self.lane_line_markings)    
    return self.lane_line_markings
         
  def white_tape_mask(self, frame=None,plot=False):
    if frame is None:
        frame = self.orig_frame
    '''''
    cv2.imshow('test', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = np.median(gray)
    #print("Mean pixel intensity:", mean)
    
    # 3. Gradient (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)
    grad_mag = cv2.convertScaleAbs(grad_mag)
    grad_mag[grad_mag < 20] = 0
    # 4. Seuillage automatique sur la luminosité
    _, light_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8) 
    light_mask = cv2.dilate(light_mask, kernel, iterations=1)

    #cv2.imshow("Light Mask", light_mask)
    # 5. Seuillage sur le gradient (élimine l’asphalte clair)
    _, grad_mask = cv2.threshold(grad_mag, 130, 255, cv2.THRESH_BINARY)
    #cv2.imshow("Gradient Mask", grad_mask)

    h, w = light_mask.shape
    mid = int(h * 0.6)
    final_mask = np.zeros_like(light_mask)
    final_mask[mid:h, :] = light_mask[mid:h, :]
    final_mask[0:mid, :] = grad_mask[0:mid, :]

    
    mask = self.crop_to_roi(final_mask)
    white_pixels = cv2.countNonZero(final_mask)
    h, w = final_mask.shape 
    total_pixels = h * w 
    ratio = white_pixels / total_pixels
    #print(f"Ratio of white pixels to total pixels: {ratio:.4f}")

    if ratio > 0.005 :
      mid = int(h * 0.75)
      final_mask = np.zeros_like(light_mask)
      final_mask[mid:h, :] = light_mask[mid:h, :]
      final_mask[0:mid, :] = grad_mask[0:mid, :]
      mask_type="Gradient Mask"
      '''''
    ''''''''''
    if np.round(ratio, 4) > 0.01:
        print("⚠️  Ratio of white pixels is high, possible false positives.")
        final_mask = grad_mask
        mask_type="Gradient Mask (Fallback)"
    
    elif np.round(ratio, 4) < 0.00049:
      print("⚠️  Ratio of white pixels is low, possible missed detections.")
      final_mask = light_mask
      mask_type="Light Mask (Fallback)"

    cv2.imshow(f"chosen : {mask_type}", final_mask)
    '''''''''
    lum = self.img.convert("L")
    avg_luminosity = np.array(lum).mean()
    #print(avg_luminosity)

    if  100 <= avg_luminosity :
      final_mask = feature_mask(frame,self.thresh_big)

    elif avg_luminosity <= 100 :
      final_mask = feature_mask(frame,self.thresh_small)

    correction = self.crop_to_roi(final_mask)
    
    h, w = correction.shape[:2]
    left_half = correction[:, :w//2]
    right_half = correction[:, w//2:]
    left_white_pixels = cv2.countNonZero(left_half)
    right_white_pixels = cv2.countNonZero(right_half)

    for i in range(5) :
      correction = self.crop_to_roi(final_mask)
      
      h, w = correction.shape[:2]
      left_half = correction[:, :w//2]
      right_half = correction[:, w//2:]
      left_white_pixels = cv2.countNonZero(left_half)
      right_white_pixels = cv2.countNonZero(right_half)
      if left_white_pixels > 200 and right_white_pixels > 200:
        break
      elif left_white_pixels < 200 or right_white_pixels < 200:
        self.thresh_big -= 30
        self.thresh_small -= 30

      if  100 <= avg_luminosity :
        final_mask = feature_mask(frame,self.thresh_big)

      elif avg_luminosity <= 100 :
        final_mask = feature_mask(frame,self.thresh_small)

    #print(f"Left half white pixels: {left_white_pixels}, Right half white pixels: {right_white_pixels}")


    if plot :
      cv2.imshow("Final Mask", final_mask)
    self.lane_line_markings = final_mask
    return  self.lane_line_markings,avg_luminosity

  def histogram_peak(self):
    """
    Get the left and right peak of the histogram
 
    Return the x coordinate of the left histogram peak and the right histogram
    peak.
    """
    midpoint = int(self.histogram.shape[0]/2)
    leftx_base = np.argmax(self.histogram[:midpoint])
    rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint
 
    # (x coordinate of left peak, x coordinate of right peak)
    return leftx_base, rightx_base
         
  def overlay_lane_lines(self, plot=True):
    """
    Overlay lane lines on the original frame
    :param: Plot the lane lines if True
    :return: Lane with overlay
    """
    # Generate an image to draw the lane lines on 
    warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       
         
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([
                         self.left_fitx, self.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([
                          self.right_fitx, self.ploty])))])
    pts = np.hstack((pts_left, pts_right))
      # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int32([pts]), (0,255, 0))
    
    cv2.polylines(color_warp, np.int32([pts_left]),  isClosed=False, color=(0, 0, 255), thickness=6)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=6)


    center_x = (self.left_fitx + self.right_fitx) / 2
    pts_center = np.array([np.transpose(np.vstack([center_x, self.ploty]))])
    
    self.pts_center_orig = cv2.perspectiveTransform(pts_center.astype(np.float32),self.inv_transformation_matrix).reshape(-1, 2)

    cv2.polylines(color_warp, np.int32([pts_center]), isClosed=False, color=(0, 0, 180), thickness=3)

    # Warp the blank back to original image space using inverse perspective 
    # matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
                                  self.orig_frame.shape[
                                  1], self.orig_frame.shape[0]))
    
    # Combine the result with the original image
    result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)
         
    if plot==True :
      
      # Plot the figures 
      figure, (ax1, ax2) = plt.subplots(2,1) # 2 rows, 1 column
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
      ax1.set_title("Original Frame")  
      ax2.set_title("Original Frame With Lane Overlay")
      plt.show()                 
    return result
  
  def perspective_transform(self, frame=None, plot=True):
    """
    Perform the perspective transform.
    :param: frame Current frame
    :param: plot Plot the warped image if True
    :return: Bird's eye view of the current lane
    """
    if frame is None:
      #frame = self.crop_to_roi(self.lane_line_markings)
      frame = self.crop_to_roi(self.lane_line_markings)
      #cv2.imshow('cropped',frame)
             
    # Calculate the transformation matrix
    self.transformation_matrix = cv2.getPerspectiveTransform(
      self.roi_points, self.desired_roi_points)
 
    # Calculate the inverse transformation matrix           
    self.inv_transformation_matrix = cv2.getPerspectiveTransform(
      self.desired_roi_points, self.roi_points)
 
    # Perform the transform using the transformation matrix
    self.warped_frame = cv2.warpPerspective(
      frame, self.transformation_matrix, self.orig_image_size, flags=(
     cv2.INTER_LINEAR)) 
 
    # Convert image to binary
    (thresh, binary_warped) = cv2.threshold(
      self.warped_frame, 127, 255, cv2.THRESH_BINARY)           
    self.warped_frame = binary_warped
 
    # Display the perspective transformed (i.e. warped) frame
    if plot == True:
      warped_copy = self.warped_frame.copy()
      warped_plot = cv2.polylines(warped_copy, np.int32([
                    self.desired_roi_points]), True, (147,20,255), 3)
 
      # Display the image
      while(1):
        cv2.imshow('Warped Image', warped_plot)
             
        # Press any key to stop
        if cv2.waitKey(0):
          break
 
      cv2.destroyAllWindows()   
             
    return self.warped_frame        
        
  def plot_roi(self, frame=None, plot=True):
    """
    Plot the region of interest on an image.
    :param: frame The current image frame
    :param: plot Plot the roi image if True
    """
    if frame is None:
      frame = self.orig_frame.copy()
    if plot == False:
      return
    elif plot == True:
    # Overlay trapezoid on the frame
      this_image = cv2.polylines(frame, np.int32([
        self.roi_points]), True, (147,20,255), 3)
      cv2.imshow('Region of Interest', this_image)

  def get_center_point_at_y(self, y_target):
      """
      Retourne le point (x_center, y_target) du centre de voie pour un y donné.
      """

      if self.left_fitx is None and self.right_fitx is None:
          print("❌ Erreur : voies pas détectées. distance laterale inconnue")
          return None,None,None,None,None

      # Trouver l'index du y le plus proche du y_target
      idx = (np.abs(self.ploty - y_target)).argmin()

      x_value_center = self.pts_center_orig[idx][0] if self.pts_center_orig is not None else None
      x_value_right = self.right_fitx[idx] if self.right_fitx is not None else None
      x_value_left = self.left_fitx[idx] if self.left_fitx is not None else None
      y_value_center = self.ploty[idx] if self.ploty is not None else None
      max_lane_y = np.max(self.ploty) if self.ploty is not None else None

      return int(x_value_center),int(y_value_center),int(x_value_left),int(x_value_right),int(max_lane_y)

  def analyze_output_images2(self,image_path,plot=False,candidates_list={}):
    #################################################### VEHICULE #############################################
    model = YOLO("yolov8x-seg.pt")
    image = cv2.imread(image_path)
    image_name = os.path.basename(image_path)
    ############# To ignore the hood of the car : mask on the bottom of the picture ################
    H, W = self.orig_frame.shape[:2]
    bottom_frac = 0.18     # adjust depending on how tall your hood looks
    y_ignore = int(H * (1 - bottom_frac))
    
    forbidden_mask = np.zeros((H, W), dtype=np.uint8)
    forbidden_mask[y_ignore:H, :] = 255
    img_for_yolo = self.orig_frame.copy()
    img_for_yolo[forbidden_mask > 0] = 0    # <-- This hides car hood from YOLO


    # voiture (2), bus (5), camion (7), moto (3), 
    results = model.predict(source=img_for_yolo, classes=[2, 5, 7, 3],verbose=False)

    '''''
    if plot:
        bev = cv2.cvtColor(self.warped_frame.copy(), cv2.COLOR_GRAY2BGR)

        # Tracer les lignes de voie UNE SEULE FOIS
        for i in range(len(self.ploty) - 1):
            xL1, y1p = int(self.left_fitx[i]), int(self.ploty[i])
            xL2, y2p = int(self.left_fitx[i+1]), int(self.ploty[i+1])
            xR1, xR2 = int(self.right_fitx[i]), int(self.right_fitx[i+1])

            cv2.line(bev, (xL1, y1p), (xL2, y2p), (0, 0, 255), 2)
            cv2.line(bev, (xR1, y1p), (xR2, y2p), (0, 0, 255), 2)
    '''''


    for r in results:
        if r.masks is None :
            continue
        for seg, box in zip(r.masks.data, r.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            '''''
            POUR CONTOUR !!!!!
            mask = seg.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            # Redimensionner le masque à la taille de l'image
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
            colored_mask = cv2.merge([np.zeros_like(mask_resized), mask_resized, np.zeros_like(mask_resized)])  # B, G, R
            image = cv2.addWeighted(image, 1.0, colored_mask, 0.0, 0)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 255), 1)
            '''
    
            color = (0, 255, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            #scv2.putText(image, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            height_px = int(y2 - y1)
            cv2.line(image, (int(x1), int(y2)), (int(x1), int(y1)), (0, 0, 255), 2)
            #cv2.putText(image,f"{height_px:.2f}", (int(x1), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
    ################################################################################################            




    ############################ calcul distance longi + lat ######################################
            
            Focal_length = 1300
            average_heights = [
                1.7,   # 0 - personne
                1.2,   # 1 - vélo
                1.60,  # 2 - voiture
                1.2,   # 3 - moto
                4.0,   # 4 - avion
                2.5,   # 5 - bus
                3.0,   # 6 - train
                3.0,   # 7 - camion 
            ]
        
            realV_height = average_heights[cls_id]
            Distance_X = ( Focal_length * realV_height / height_px )
            distance_text = f"{Distance_X:.1f}"
            cv2.putText(image, distance_text, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            

            x_centre = (x1+x2)/2
            image_width = image.shape[1]
            c_x,y_value_center,nearest_x_left, nearest_x_right,max_lane_y = self.get_center_point_at_y(int(y1))
            X_lat = (x_centre - c_x) * (Distance_X) / Focal_length  # en mètres
            #print(X_lat)
            cv2.putText(image, f"{X_lat:.2f}", (int(x2), int(y2) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            
            if image_name not in candidates_list:
                candidates_list[image_name] = []
            candidates_list[image_name].append({
                "distance_longitudinale": Distance_X,
                "distance_laterale": X_lat,
                "nearest_left" : nearest_x_left,
                "nearest_right" : nearest_x_right,
                "nearest_y" : y_value_center,
                "y_bottom" : y1,
                "y_top" : y2,
                "x_left" : x1,
                "x_right" : x2,
                "max_lane_y" : max_lane_y
            })
            ############# tester une autre méthode d'analyse pour voir si le véhicule est dans la voie ##############
            
    cv2.imwrite(image_path, image)
    return candidates_list
    """""""""
                M_vehicle = cv2.getPerspectiveTransform(self.roi_points, self.desired_roi_points)
                bev_vec = cv2.warpPerspective(self.orig_frame,M_vehicle,self.orig_image_size)
                #cv2.imshow('vec bev', bev_vec)
                # Projetter le bbox YOLO → BEV
                pts_img = np.float32([
                    [x1, y2],
                    [x2, y2]
                ]).reshape(-1, 1, 2)

                pts_bev = cv2.perspectiveTransform(pts_img, M_vehicle)



    #############
                if plot:

                    # 1) base BEV = warped_frame
                    bev = cv2.cvtColor(self.warped_frame.copy(), cv2.COLOR_GRAY2BGR)

                    # 2) tracer les lanes BEV (left_fitx / right_fitx)
                    for i in range(len(self.ploty) - 1):
                        xL1, y1 = int(self.left_fitx[i]), int(self.ploty[i])
                        xL2, y2 = int(self.left_fitx[i+1]), int(self.ploty[i+1])
                        xR1 = int(self.right_fitx[i])
                        xR2 = int(self.right_fitx[i+1])

                        cv2.line(bev, (xL1, y1), (xL2, y2), (0, 0, 255), 2)  # lane gauche
                        cv2.line(bev, (xR1, y1), (xR2, y2), (0, 0, 255), 2)  # lane droite

                    # 3) dessiner le bbox en BEV
                    xb1 = int(pts_bev[0][0][0])
                    xb2 = int(pts_bev[1][0][0])
                    yb1 = int(pts_bev[0][0][1])
                    yb2 = int(pts_bev[1][0][1])

                    # couleur : vert si in_lane_bev existe ET True
                    color = (0,255,0) if candidates_list[image_name][-1].get("in_lane_bev", False) else (0,0,255)

                    cv2.rectangle(bev, (xb1, yb1), (xb2, yb2), color, 2)
                    cv2.imshow('image',image)
                    cv2.imshow("BEV minimaliste", bev)
                    cv2.waitKey(1)




                '''''
                if plot :
                  cv2.imshow('Image with bounding boxes', image)
                '''''
        ##########################################################################################################
    """""""""""""""
  
  def validate_lanes(self, curr_leftx,curr_rightx,road_shape):
    LANE_WIDTH_PX =700
    corrected_by_mean_value_method = False
    lane_width = curr_rightx - curr_leftx
    somme =0 
    for value in lane_width :
        somme += value
    mean_lane_width = somme / len(lane_width)
    if abs(mean_lane_width) < 80 :
        curr_leftx -= 250
        curr_rightx += 250
        corrected_by_mean_value_method =True
    
    too_close = np.any(lane_width<150)
    too_big = np.any(lane_width>700)
    crossing= np.any(curr_leftx>curr_rightx)
    curve_too_sharp_straight = self.left_curvem<10000 or self.right_curvem<10000
    curve_too_sharp_slight = self.left_curvem<5000 or self.right_curvem<5000
    shape_is_straight = road_shape.lower() in ["straight"]
    shape_is_slightly_curved = road_shape.lower() in ["slight_curve_right", "slight_curve_left"]
  


    if (too_close or crossing or (curve_too_sharp_straight and shape_is_straight) or (curve_too_sharp_slight and shape_is_slightly_curved) or too_big) and not corrected_by_mean_value_method:
      for index, value in enumerate(lane_width) :
        curr_leftx[index] = self.width/2 - LANE_WIDTH_PX/2 
        curr_rightx[index] = self.width/2 + LANE_WIDTH_PX/2
    else :
       self.passed_validation_test = True

      
    self.left_fitx = curr_leftx
    self.right_fitx = curr_rightx
              
    return curr_leftx,curr_rightx

  def create_fictive_lanes(self):
    """
    Crée des voies fictives quand aucune voie n'est détectée.
    Utilise le centre de l'image et une largeur de voie standard.
    
    :param road_shape: La forme de la route (straight, slight_curve_right, etc)
    :return: left_fitx, right_fitx, left_fit, right_fit, ploty
    """
    LANE_WIDTH_PX = 500
    
    # Créer les y-coordinates (ploty)
    ploty = np.linspace(0, self.warped_frame.shape[0], self.warped_frame.shape[0])
    
    # Centre horizontal de l'image
    center_x = self.warped_frame.shape[1] / 2
    
    # Créer les positions x fictives (lignes droites parallèles)
    left_fitx = np.ones_like(ploty) * (center_x - LANE_WIDTH_PX / 2)
    right_fitx = np.ones_like(ploty) * (center_x + LANE_WIDTH_PX / 2)
    
    # Créer les coefficients polynomiaux (fit) pour ces lignes droites
    # Pour une ligne droite : y = a*x^2 + b*x + c, on a [0, 0, c]
    left_fit = np.array([0, 0, center_x - LANE_WIDTH_PX / 2])
    right_fit = np.array([0, 0, center_x + LANE_WIDTH_PX / 2])
    
    # Définir les variables self pour le reste du pipeline
    self.ploty = ploty
    self.left_fitx = left_fitx
    self.right_fitx = right_fitx
    self.left_fit = left_fit
    self.right_fit = right_fit
    
    # Initialiser aussi les variables de pixels pour le calcul de courbure
    self.lefty = ploty
    self.righty = ploty
    self.leftx = left_fitx
    self.rightx = right_fitx
    self.left_curvem = 0
    self.right_curvem = 0
    self.center_offset = 0
    
   #print(f"✅ Voies fictives créées : largeur={LANE_WIDTH_PX}px, centre={center_x:.0f}px")
    
    return left_fitx, right_fitx, left_fit, right_fit, ploty

  def lane_detection_quality(self):
    quality_score = 0
    
    # 1. Vérifier si les voies ont été détectées (pas fictives)
    if self.detected_lanes:
        quality_score += 40  # Base score pour détection réelle
    else:
        return 20  # Pas de détection = score 0
    
    if self.numberof_lanesdetected == 1 :
        quality_score += 15  # Pénalité pour une seule voie détectée
    elif self.numberof_lanesdetected == 2 :
        quality_score += 30  # Bonus pour deux voies détectées
    
    if self.passed_validation_test :
        quality_score += 20  # Bonus pour passage du test de validation

    return int(np.clip(quality_score, 0, 100))

def get_info_from_excel( excel_file,subfolder=None):
    """
    Compare the detected lane lines to the ground truth data from Texcel.
    :param: texcel_data The ground truth data from Texcel
    :return: Comparison results
    """
    # This function would contain code to compare the detected lane lines
    # to the ground truth data from Texcel and return the results of the comparison.   
    df = pd.read_excel(excel_file)
    
    pattern = r"event_(\d{8})_(\d{6})_instant(\d+\.?\d*)"
    match = re.search(pattern, subfolder)

    if not match:
        raise ValueError(f"Format invalide : {subfolder}. Le format attendu est 'event_YYYYMMDD_HHMMSS_instantX.XX'.")

    date_str = match.group(1)
    time_str = match.group(2)
    instant = float(match.group(3))
    for index, row in df.iterrows():
      try :
        instant_raw = row['Debut event en DareDeevil(approxiamatif)']
      
        if (instant == instant_raw) and (f"{date_str}_{time_str}" in row['Nom du log']):
            distance_lat_excel = float(row['V_m_TargetYdistMeas apres ']) if row['V_m_TargetYdistMeas apres '] != "-" else None
            distance_long_excel = float(row['V_m_Distance_Meas apres ']) if row['V_m_Distance_Meas apres '] != "-" else None
            event_truth = (row['Event status with automatic analysis']) if row['Event status with automatic analysis'] != "-" else None
            reason_excel = (row['Reason']) if row['Reason'] != "-" else None
            TTC = float(row['TTC(s)']) if row['TTC(s)'] != "-" else None
            duration = float(row['Duree event']) if row['Duree event'] != "-" else None
            category = category_calculator(TTC, duration)
            return distance_long_excel, distance_lat_excel,event_truth,reason_excel,category
      except Exception as e:
          print(f"Error processing row {index}: {e}")
          continue

def get_ego_id_from_excel(excel_file, subfolder=None):
        """
        Compare the detected lane lines to the ground truth data from Texcel.
        :param: texcel_data The ground truth data from Texcel
        :return: Comparison results
        """
        # This function would contain code to compare the detected lane lines
        # to the ground truth data from Texcel and return the results of the comparison.   
        df = pd.read_excel(excel_file)
        
        pattern = r"event_(\d{8})_(\d{6})_instant(\d+\.?\d*)"
        match = re.search(pattern, str(subfolder))

        if not match:
            raise ValueError(f"Format invalide (main.py): {subfolder}. Le format attendu est 'event_YYYYMMDD_HHMMSS_instantX.XX'.")

        date_str = match.group(1)
        time_str = match.group(2)
        instant = float(match.group(3))
        for index, row in df.iterrows():
            try :
                instant_raw = row['Debut event en DareDeevil(approxiamatif)']
            
                if (instant == instant_raw) and (f"{date_str}_{time_str}" in row['Nom du log']):
                    ego_id = row['Nom du log'].split('_')[2]
                    return ego_id
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue

def category_calculator(TTC,duration):
  category = None
  if duration >= 0.2 and TTC <= 5 :
    category = 1
  if duration >= 0.2 and 5 < TTC <= 10 :
    category = 2
  if duration >= 0.1 and category!=1 and category!= 2 :
    category = 3
  if duration <= 0.1 :
    category = 4
  return category

def get_info_from_excel_multi_sheets(excel_file, output_folder="output", subfolder=None):
    """
    Lit toutes les sheets de l'excel, cherche la ligne correspondant à subfolder
    (pattern event_YYYYMMDD_HHMMSS_instantX.XX) et retourne
    (distance_long_excel, distance_lat_excel, event_truth, reason_excel)
    ou (None, None, None, None) si non trouvé.
    """
    import pandas as pd
    import re

    # lire toutes les sheets
    xls = pd.read_excel(excel_file, sheet_name=None)

    pattern = r"event_(\d{8})_(\d{6})_instant(\d+\.?\d*)"
    match = re.search(pattern, str(subfolder))
    if not match:
        raise ValueError(f"Format invalide : {subfolder}. Le format attendu est 'event_YYYYMMDD_HHMMSS_instantX.XX'.")

    date_str = match.group(1)
    time_str = match.group(2)
    instant = float(match.group(3))

    # noms de colonnes attendus (conserver ceux qui sont utilisés ailleurs)
    col_instant = "Debut event en DareDeevil(approxiamatif)"
    col_nom = "Nom du log"
    col_lat = "V_m_TargetYdistMeas apres "
    col_long = "V_m_Distance_Meas apres "
    col_type = "remarks"
    col_comment = "Remarks"

    for sheet_name, df in xls.items():
        if not isinstance(df, pd.DataFrame):
            continue
        # skip sheet si elle n'a pas les colonnes minimales
        if col_instant not in df.columns or col_nom not in df.columns:
            continue

        for idx, row in df.iterrows():
            try:
                instant_raw = row[col_instant]
            except Exception:
                # cellule manquante/format inattendu -> ignorer la ligne
                continue

            # protéger contre les NaN / types inattendus
            try:
                if pd.isna(instant_raw):
                    continue
                instant_raw_f = float(instant_raw)
            except Exception:
                continue

            if instant_raw_f == instant and f"{date_str}_{time_str}" in str(row[col_nom]):
                # extraire en défensive
                def safe_get(key):
                    if key in df.columns:
                        val = row[key]
                        return None if pd.isna(val) else val
                    return None

                distance_lat_excel = safe_get(col_lat)
                distance_long_excel = safe_get(col_long)
                event_truth = safe_get(col_type)
                reason_excel = safe_get(col_comment)

                try:
                    distance_lat_excel = float(distance_lat_excel) if distance_lat_excel not in (None, "-") else None
                except Exception:
                    distance_lat_excel = None
                try:
                    distance_long_excel = float(distance_long_excel) if distance_long_excel not in (None, "-") else None
                except Exception:
                    distance_long_excel = None

                return distance_long_excel, distance_lat_excel, event_truth, reason_excel

    # rien trouvé → renvoyer tuple vide (évite le TypeError lors de l'unpacking)
    return None, None, None, None

def identify_car(distance_long_excel, distance_lat_excel,car_candidates):
    excel_longi = distance_long_excel
    excel_lat = distance_lat_excel
    yolo_detections = car_candidates

    max_error = 17
    best_det = None
    best_err = float("inf")
    if yolo_detections:
      for det in yolo_detections:
          #print(det)
          err = ((det["distance_longitudinale"] - excel_longi)**2 +2*((-det["distance_laterale"] - excel_lat)**2))**0.5
          #print(err)
          #print(err)
          #print(excel_longi, excel_lat)
          if err < best_err:
              best_err = err
              best_det = det
      #print('best error',best_err)
    
    else : 
       best_err = 0

    if best_det is None:
        return best_det,best_err

    # Reject if too far
    if best_err > max_error:
        return None,best_err
    
    return best_det,best_err

def add_to_final_list(chosen_dict, frame_key, match_dict):
  if match_dict:
    chosen_dict[frame_key] = {
        "distance_longitudinale": match_dict["distance_longitudinale"],
        "distance_laterale": match_dict["distance_laterale"],
          "nearest_left"   : match_dict["nearest_left"],
          "nearest_right"  : match_dict["nearest_right"],
          "nearest_y"      : match_dict["nearest_y"],
          "y_bottom"       : match_dict["y_bottom"],
          "y_top"          : match_dict["y_top"],
          "x_left"         : match_dict["x_left"],
          "x_right"        : match_dict["x_right"],
          "max_lane_y"     : match_dict["max_lane_y"]
    }

def score_distance(longi0, max_dist=100):
    """
    Score distance entre 0 et 1 : 1 = très proche, 0 = très loin.
    Décroissance linéaire (continuité, pas de paliers).
    """
    return np.clip(1 - (longi0 / max_dist), 0, 1)

def score_visibility(max_lane_y, y_bottom, k=40):
    """
    Score entre 0 et 1 basé sur la visibilité.
    Utilise une fonction sigmoïde pour éviter les sauts brusques.
    """
    return 1 / (1 + np.exp(-(max_lane_y - y_bottom) / k))

def score_road_shape(road_shape):
    """
    Score facteur route selon la courbure.
    """
    mapping = {
        'straight'            : 1.0,
        'slight_curve_left'   : 0.9,
        'slight_curve_right'  : 0.9,
        'right_curve'         : 0.6,
        'left_curve'          : 0.6
    }
    return mapping.get(road_shape, 1.0)

def compute_trust_score(longi0, max_lane_y, y_bottom, road_shape): #lane_quality_score=100):
    """
    Combinaison pondérée des 3 facteurs continus.
    Retourne un score final entre 0 et 100.
    """

    sd   = score_distance(longi0)                    # 50% du score
    sv   = score_visibility(max_lane_y, y_bottom)    # 25% du score
    sr   = score_road_shape(road_shape)              # 10% du score
    #sq   = lane_quality_score / 100.0                # 15% du score (normalisé 0-1)

    trust = 100 * (0.6 * sd + 0.25 * sv + 0.15 * sr )#+ 0.15 * sq)
    return int(np.clip(trust, 0, 100))

def overselection(selected_cars, fichier_txt, road_shape):#, lane_quality_score=100):
  trust_score = 0
  longi = 0
  lat0 = 0
  found = False
  if selected_cars :
    for photo in selected_cars :
      if 'tplus0.0' in photo:
          found = True
          longi0 = selected_cars[photo]['distance_longitudinale']
          lat0 = selected_cars[photo]['distance_laterale']
          nearest_y      = selected_cars[photo]["nearest_y"]
          max_lane_y     = selected_cars[photo]["max_lane_y"]
          y_bottom          = selected_cars[photo]["y_bottom"]
         
          trust_score = compute_trust_score(longi0, max_lane_y, y_bottom, road_shape)#, lane_quality_score)
    if not found :
      for photo in selected_cars :
        longi0 = selected_cars[photo]['distance_longitudinale']
        lat0 = selected_cars[photo]['distance_laterale']
        nearest_y      = selected_cars[photo]["nearest_y"]
        max_lane_y     = selected_cars[photo]["max_lane_y"]
        y_bottom          = selected_cars[photo]["y_bottom"]
        trust_score = compute_trust_score(longi0, max_lane_y, y_bottom, road_shape) #, lane_quality_score)
        break
    
    final = "False event"
    cut_out = False
    cut_in = False
    from_inlane = False
    # Cas simple : dans la même voie → False overselection
    if abs(lat0) <= 2:
      t = -1.5
      while t <= -0.5:
          key = f"tminus{abs(t):.1f}s"
          if key in selected_cars:  
            info           = selected_cars[key]
            dist_longi     = info["distance_longitudinale"]
            dist_lat       = info["distance_laterale"]
            nearest_left   = info["nearest_left"]
            nearest_right  = info["nearest_right"]
            nearest_y      = info["nearest_y"]
            y_bottom       = info["y_bottom"]
            y_top          = info["y_top"]
            x_left         = info["x_left"]
            x_right        = info["x_right"]
            max_lane_y     = info["max_lane_y"]

            if abs(dist_lat) > 2 :
                cut_in = True
            #if abs(lat) > 2 and abs(lat0) < 2 :
            #   cut_in = True 
          t += 0.5
      t = 0.5
      while t <= 1.5 and not cut_in:
          key = f"tplus{t:.1f}s"
          if key in selected_cars:
            info           = selected_cars[key]
            dist_longi     = info["distance_longitudinale"]
            dist_lat       = info["distance_laterale"]
            nearest_left   = info["nearest_left"]
            nearest_right  = info["nearest_right"]
            nearest_y      = info["nearest_y"]
            y_bottom       = info["y_bottom"]
            y_top          = info["y_top"]
            x_left         = info["x_left"]
            x_right        = info["x_right"]
            max_lane_y     = info["max_lane_y"]
            if abs(dist_lat) >= 2 :
                cut_out = True
          t += 0.5
      if not cut_in and not cut_out:
        from_inlane = True
        reason ="Target in Lane"
        final = "False event"
    
    else: 
      # ---- 3a) Regarder avant (de -2.0s à -0.5s, pas de 0.5s) ----
      t = -1.5
      while t <= -0.5:
          key = f"tminus{abs(t):.1f}s"
          if key in selected_cars:
              
            info           = selected_cars[key]
            dist_longi     = info["distance_longitudinale"]
            dist_lat       = info["distance_laterale"]
            nearest_left   = info["nearest_left"]
            nearest_right  = info["nearest_right"]
            nearest_y      = info["nearest_y"]
            y_bottom       = info["y_bottom"]
            y_top          = info["y_top"]
            x_left         = info["x_left"]
            x_right        = info["x_right"]
            max_lane_y     = info["max_lane_y"]

            if abs(dist_lat) <= 2 :
                cut_out = True
            #if abs(lat) > 2 and abs(lat0) < 2 :
            #   cut_in = True 
          t += 0.5


      # ---- 3b) Regarder après (de +0.5s à +2.0s) ----
      t = 0.5
      while t <= 1.5 and not cut_out:
          key = f"tplus{t:.1f}s"
          if key in selected_cars:
            info           = selected_cars[key]
            dist_longi     = info["distance_longitudinale"]
            dist_lat       = info["distance_laterale"]
            nearest_left   = info["nearest_left"]
            nearest_right  = info["nearest_right"]
            nearest_y      = info["nearest_y"]
            y_bottom       = info["y_bottom"]
            y_top          = info["y_top"]
            x_left         = info["x_left"]
            x_right        = info["x_right"]
            max_lane_y     = info["max_lane_y"]
            if abs(dist_lat) < 2 and abs(lat0) > 2:
                cut_in = True
          t += 0.5

    # ---- 4) Résultats ----
    if cut_out:
        reason = "Cut Out"
        final = "False event"

    if cut_in:
        reason = "Cut In"
        final = "False event"

    if not cut_out and not cut_in and not from_inlane:
        reason = "No cut-in/out detected, target not in lane"
        final = "True event"
  
  else :
     reason = 'No car detection. Requires manual check'
     final = 'No analysis, car not found'
  #fichier_txt.write(reason,'trust_score :',trust_score)
  return reason, trust_score,final

def write_event_status_to_excel(excel_path, excel_script):

  try:
      df_gt = pd.read_excel(excel_path)
      df2 = pd.read_excel(excel_script)
  except Exception as e:
      if df2.empty:
        raise RuntimeError(f"Cannot read Excel file {excel_script}: {e}")
      elif df_gt.empty:
        raise RuntimeError(f"Cannot read Excel file {excel_path}: {e}")

  #try:
  cols_gt = list(df_gt.columns)
  target_col = None
  for name in cols_gt:
      if str(name).strip().lower() in ("event status","event status with automatic analysis"):
          target_col = name
          break

  insert_pos = cols_gt.index(target_col) + 1 if target_col else None
  new_col = "event status AI"
  if new_col not in df_gt.columns and insert_pos!=None:
      df_gt.insert(insert_pos, new_col, "")
    

  for index1, row1 in df2.iterrows():
    subfolder = row1['Subfolder']
    event_script = row1['event_truth_script']

    pattern = r"event_(\d{8})_(\d{6})_instant(\d+\.?\d*)"
    match = re.search(pattern, subfolder)

    if not match:
        raise ValueError(f"Format invalide : {subfolder}. Le format attendu est 'event_YYYYMMDD_HHMMSS_instantX.XX'.")

    date_str = match.group(1)
    time_str = match.group(2)
    instant = float(match.group(3))

    for index, row in df_gt.iterrows():
      instant_raw = row['Debut event en DareDeevil(approxiamatif)']
    
      if (instant == instant_raw) and (f"{date_str}_{time_str}" in row['Nom du log']):
          df_gt.at[index, new_col] = event_script
          break
    
    try:
      new_excel_path = excel_path.replace(".xlsx", "_updated.xlsx")
      df_gt.to_excel(new_excel_path, index=False)
    except Exception as e: 
      raise RuntimeError(f"Cannot write to Excel file {excel_path}: {e}")
  #except Exception as e:
   # print(f"Error occurred while adding AI script column in the intiial excel sheet: {e}")

def debug_plotting_mode(summary_rows,summary_all ,output_folder,category=None):
   ################# SAVING TO EXCEL THE SUMMARY OF THE ANALYSIS ####################
    summary_df = pd.DataFrame(summary_all, columns=["Subfolder", "road_shape","reason_script","reason_excel","event_truth_script","event_truth_excel","Trust_score","lowest_matching_error","category"])

    #total_subfolders = len(summary_rows)
    #summary_df.loc[len(summary_df)] = ["Number of events", total_subfolders," "," "," "," "," "," "]
    moyenne, score_petit ,counter,nocardetection,correct_analysis,correct_analysis_over_70= 0, 0, 0, 0, 0,0

    
    
    # Save initial Excel file first
    summary_file = output_folder / "Summary1.xlsx"
    if not summary_file.exists():
      summary_df.to_excel(summary_file, index=False)
    else:
       wb = load_workbook(summary_file) 
       
    
    # ==================== IMPROVED THRESHOLD ANALYSIS ====================
    # Calculate Precision, Recall, F1-score, and FPR for each threshold
    thresholds = list(range(1, 91))
    metrics_by_threshold = []
    capsules_per_threshold = []
    trust_scores = [row[6] for row in summary_rows]
    
    
    for threshold in thresholds:
        TP = 0  # Correctly detected cut-in/cut-out (script=Excel AND script detected event)
        FP = 0  # False alarm (script detected event BUT Excel says no event)
        FN = 0  # Missed event (script missed BUT Excel says there's an event)
        TN = 0  # Correctly rejected (script=Excel AND no event)
        counter = 0
        for i in range(len(summary_rows)):
            trust_score = summary_rows[i][6]
            script_event = summary_rows[i][4]  # event_truth_script
            excel_event = summary_rows[i][5]   # event_truth_excel
            script_reason = summary_rows[i][2]
            excel_reason = summary_rows[i][3]
            counter += 1 if trust_score > 0 else 0

            # Only count if trust score >= threshold
            if trust_score >= threshold and excel_event != 0 and excel_event != None:
                if (script_event.lower() == excel_event.lower() or script_event.lower() in excel_event.lower()):
                    if script_event.lower() == "true event":  # Both say "Overselection"
                        TP += 1
                    elif script_event.lower() == "false event":  # Both say "Normal"
                        TN += 1
                else:
                    if script_event.lower() == "true event" and excel_event.lower() == "false event":
                        FP += 1  # False alarm
                    elif script_event.lower() == "false event" and excel_event.lower() == "true event":
                        FN += 1  # Missed detection

            else:
                continue
        
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        sample_count = TP + FP + FN + TN
        fpr = FP / sample_count if sample_count > 0 else 0  
        fnr = FN / sample_count if sample_count > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        coverage = (sample_count / counter)*100 if counter > 0 else 0
        success_percentage = ((TN+TP)/sample_count)*100 if sample_count > 0 else 0 
        metrics_by_threshold.append({
        'threshold': threshold,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'specificity': specificity,
        'npv': npv,
        'coverage': coverage,
        'sample_count': sample_count,
        'success_percentage': success_percentage,
        'fnr' : fnr
         })

        capsules_per_threshold.append(sum(1 for score in trust_scores if score == threshold))
    # ============================
    #  WRITE METRICS TO EXCEL
    # ============================

    wb = load_workbook(summary_file)

    if f"CAT{category}" in wb.sheetnames:
        ws = wb[f"CAT{category}"]
    else:
        ws = wb.create_sheet(f"CAT{category}")

    ws['A1'] = 'Threshold'
    ws['B1'] = 'TP'
    ws['C1'] = 'FP'
    ws['D1'] = 'FN'
    ws['E1'] = 'TN'
    ws['F1'] = 'Precision'
    ws['G1'] = 'Recall'
    ws['H1'] = 'F1'
    ws['I1'] = 'FPR'
    ws['J1'] = 'Accuracy'
    ws['K1'] = 'Sample Count'
    ws['L1'] = 'Specificity'
    ws['M1'] = 'NPV'
    ws['N1'] = 'Coverage'
    ws['O1'] = 'success_percentage'
    ws['P1'] = 'fnr'

    for idx, m in enumerate(metrics_by_threshold, start=2):
        ws[f'A{idx}'] = m['threshold']
        ws[f'B{idx}'] = m['TP']
        ws[f'C{idx}'] = m['FP']
        ws[f'D{idx}'] = m['FN']
        ws[f'E{idx}'] = m['TN']
        ws[f'F{idx}'] = round(m['precision'], 3)
        ws[f'G{idx}'] = round(m['recall'], 3)
        ws[f'H{idx}'] = round(m['f1_score'], 3)
        ws[f'I{idx}'] = round(m['fpr'], 3)
        ws[f'J{idx}'] = round(m['accuracy'], 3)
        ws[f'K{idx}'] = m['sample_count']
        ws[f'L{idx}'] = round(m['specificity'], 3)
        ws[f'M{idx}'] = round(m['npv'], 3)
        ws[f'N{idx}'] = round(m['coverage'], 3)
        ws[f'O{idx}'] = round(m['success_percentage'], 3)
        ws[f'P{idx}'] = round(m['fnr'], 3)


    # ============================
    #  PLOTS (FPR, Coverage, Success %)
    # ============================

    thresholds_list = [m['threshold'] for m in metrics_by_threshold]
    fprs = [m['fpr'] for m in metrics_by_threshold]
    fnrs = [m['fnr'] for m in metrics_by_threshold]
    coverage_list = [m['coverage'] for m in metrics_by_threshold]
    success_list = [m['success_percentage'] for m in metrics_by_threshold]

    plt.figure(figsize=(14, 10))

    # 1. FPR
    plt.subplot(5, 1, 1)
    plt.plot(thresholds_list, fprs, 'r-', linewidth=2)
    plt.title("False Positive Rate (FPR) vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("FPR")
    plt.grid(True, alpha=0.3)

    # 2. Coverage
    plt.subplot(5, 1, 2)
    plt.plot(thresholds_list, coverage_list, 'b-', linewidth=2)
    plt.title("Coverage vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Coverage (%)")
    plt.grid(True, alpha=0.3)

    # 3. Success %
    plt.subplot(5, 1, 3)
    plt.plot(thresholds_list, success_list, 'g-', linewidth=2)
    plt.title("Success Percentage (TP+TN) vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Success (%)")
    plt.grid(True, alpha=0.3)

    # 4. FNR
    plt.subplot(5, 1, 4)
    plt.plot(thresholds_list, fnrs, 'b-', linewidth=2)
    plt.title("False Negative Rate (FNR) vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("FNR")
    plt.grid(True, alpha=0.3)

    plt.subplot(5, 1, 5)
    plt.scatter(thresholds_list, capsules_per_threshold, c='purple', s=40)
    plt.title("Nuage de points : Nombre de capsules par threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Capsules")
    plt.grid(True, alpha=0.3)


    plt.tight_layout()
    plot_path = f"threshold_analysis_CAT{category}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


    # Insert plot into Excel
    img = ExcelImage(plot_path)
    ws.add_image(img, "A20")

    wb.save(summary_file)

    return summary_file



def calculer_chiffres_cles(summary_file,threshold_CAT1=70,threshold_CAT2=50,threshold_CAT3=30,threshold_CAT4=10):
    wb = pd.read_excel(summary_file)

    # Compteurs globaux
    total_events = 0
    true_events = 0
    false_events = 0
    not_analysable = 0
    above_min_trust = 0

    # Compteurs par catégorie
    cat_counts = {
        1: {"total": 0, "done": 0, "not_done": 0},
        2: {"total": 0, "done": 0, "not_done": 0},
        3: {"total": 0, "done": 0, "not_done": 0},
        4: {"total": 0, "done": 0, "not_done": 0},
    }

    thresholds = {
        1: threshold_CAT1,
        2: threshold_CAT2,
        3: threshold_CAT3,
        4: threshold_CAT4,
    }

    for _, row in wb.iterrows():
        total_events += 1
        cat = row['category']
        trust_score = row['Trust_score']
        event_truth = row['event_truth_script'] 
        reason = row['reason_script']

        if event_truth == "True event":
            true_events += 1
        elif event_truth == "False event":
            false_events += 1

        # Capsule non analysable
        if reason == "No car detection. Requires manual check":
            not_analysable += 1
            continue

        # Comptage par catégorie
        if cat in cat_counts:
            cat_counts[cat]["total"] += 1

            if trust_score >= thresholds[cat]:
                cat_counts[cat]["done"] += 1
                above_min_trust += 1
            else:
                cat_counts[cat]["not_done"] += 1

            

    return {
        "total_events": total_events,
        "true_events": true_events,
        "false_events": false_events,
        "non_analysables": not_analysable,
        "above_min_trust_score": above_min_trust,
        "capsules_par_categorie": {
            "CAT1": cat_counts[1],
            "CAT2": cat_counts[2],
            "CAT3": cat_counts[3],
            "CAT4": cat_counts[4],
        }
    }





def generate_report(summary_file, output_folder, image_path, analysis_name):
    # Calcul des chiffres clés
    stats = calculer_chiffres_cles(summary_file)

    total_events = stats["total_events"]
    true_events = stats["true_events"]
    false_events = stats["false_events"]
    non_analysables = stats["non_analysables"]
    above_min_trust = stats["above_min_trust_score"]
    cats = stats["capsules_par_categorie"]

    # Création du PDF
    pdf_path = os.path.join(output_folder, "rapport_chiffres_cles.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)

    width, height = A4
    y = height - 50

    # ----- IMAGE EN HAUT À DROITE -----
    if image_path and os.path.exists(image_path):
        img_width = 10 * cm
        img_height = 10 * cm
        c.drawImage(
            image_path,
            width - img_width - 50,
            height - img_height - 30,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True
        )

    # ----- NOM DE L’ANALYSE -----
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, analysis_name)

    y -= 15
    text_width = c.stringWidth(analysis_name, "Helvetica-Bold", 16)
    c.line(50, y, 50 + text_width, y)
    y -= 40

    # ----- TITRE -----
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, "Rapport – Chiffres Clés")
    y -= 40

    # ----- INFOS GLOBALES -----
    c.setFont("Helvetica", 12)

    c.drawString(50, y, f"Nombre total d’events : {total_events}")
    y -= 20

    c.drawString(50, y, f"True events : {true_events}")
    y -= 20

    c.drawString(50, y, f"False events : {false_events}")
    y -= 20

    c.drawString(50, y, f"Capsules non analysables : {non_analysables}")
    y -= 20

    c.drawString(
        50,
        y,
        f"Capsules au-dessus du seuil de confiance : {above_min_trust}"
    )
    y -= 40

    # ----- DÉTAIL PAR CATÉGORIE -----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Détail par catégorie")
    y -= 30

    c.setFont("Helvetica", 12)

    for cat in ["CAT1", "CAT2", "CAT3", "CAT4"]:
        total_cat = cats[cat]["total"]
        done = cats[cat]["done"]
        not_done = cats[cat]["not_done"]

        c.drawString(
            50,
            y,
            f"{cat} : Total = {total_cat} | above minimum trust score = {done} | below minimum trust score = {not_done}"
        )
        y -= 20

        # Saut de page si nécessaire
        if y < 100:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 50

    # ----- CONFIDENTIEL -----
    c.setFont("Helvetica-Bold", 10)
    confidential_text = "CONFIDENTIEL"
    text_width = c.stringWidth(confidential_text, "Helvetica-Bold", 10)

    c.drawString(
        (width - text_width) / 2,
        30,
        confidential_text
    )

    c.save()
    return pdf_path



def main(filename=r"C:\Users\az03810\OneDrive - Alliance\Bureau\ia_detection\ia_detection\output\20221003_110255\event_20221003_110255_instant69.69_offset0.9218347448363602\20221003_110255_tminus1.5s_slight_curve_left.jpg"):
     
  original_frame = cv2.imread(filename)
 
  lane_obj = Lane(orig_frame=original_frame, img_path=filename,road_shape='straight')
  lane_line_markings,avg_lum = lane_obj.white_tape_mask(plot=False)
 
  cv2.imshow("Image", lane_line_markings)
  cv2.waitKey(0)  
  lane_obj.plot_roi(plot=True)

  warped_frame = lane_obj.perspective_transform(plot=False)
  histogram = lane_obj.calculate_histogram(plot=False)  
  
  left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(
    plot=False)
 
  curr_left_fit, curr_right_fit,curr_left_fitx, curr_right_fitx, curr_ploty =lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
  lane_obj.calculate_curvature(print_to_terminal=False)
  #print(lane_obj.left_curvem,lane_obj.right_curvem)
  curr_left_fitx, curr_right_fitx= lane_obj.validate_lanes(curr_left_fitx, curr_right_fitx,road_shape='straight')
     
  frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=True)
 
  
  print(lane_obj.left_curvem,lane_obj.right_curvem)
                                                               
  lane_obj.calculate_car_position(print_to_terminal=False)
     
  frame_with_lane_lines2 = lane_obj.display_curvature_offset(
    frame=frame_with_lane_lines, plot=True)
  cv2.waitKey(0) 
  cv2.destroyAllWindows() 

#main()
