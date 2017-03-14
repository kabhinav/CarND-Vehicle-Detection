
# coding: utf-8

# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---
# ##  camera calibration using chessboard images

# In[1]:

import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')


M = pickle.load( open('M.p', 'rb') )
Minv = pickle.load( open('Minv.p', 'rb') )
mtx = pickle.load( open('mtx.p', 'rb') )
dist = pickle.load( open('dist.p', 'rb') )

def threshold_image(img, thresh=(0, 255)):
    thresholded = img.copy()
    thresholded[thresholded <= thresh[0]] = 0
    thresholded[thresholded > thresh[1]] = 0
    return thresholded

# Thresholding channel s from a hls image
def get_S(img):
    thresh_s = (100, 255)
    hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) 
    return threshold_image(hls_img[:,:,2], thresh=thresh_s)

def get_binary(img):
    binary = np.zeros_like(img)
    binary[img > 0] = 1
    return binary

# Threshold channel R
def get_R(img):
    thresh_r = (190, 255)
    return threshold_image(img[:,:,0], thresh=thresh_r)

# Combine channel R and S
def combine_rs(r_img, s_img):
    combined_img = r_img.astype(int)
    combined_img = combined_img + s_img
    combined_img = (combined_img/combined_img.max()) * 255
    # Convert to binary
    binary = get_binary(threshold_image(combined_img, thresh=(110,255)))
    return binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients                                                                                                                                                                                                       
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude                                                                                                                                                                                                        
    gradmag = np.sqrt(sobelx**2 + sobely**2)                                                                                                                                                                                                                
    scaled_mag = np.uint8(255*gradmag/np.max(gradmag))
    # Create a binary image of ones where threshold is met, zeros otherwise                                                                                                                                                                   
    binary_output = scaled_mag.copy()
    binary_output[scaled_mag <= thresh[0]] = 0
    binary_output[scaled_mag > thresh[1]] = 0
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = absgraddir.copy()
    binary_output[absgraddir <= thresh[0]] = 0
    binary_output[absgraddir > thresh[1]] = 0
    return binary_output

def combine_grad_thresholds(img, thresh=(110, 255)):
    img = np.copy(img)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    combined_rs = combine_rs(get_R(img), get_S(img))
    gradient_mag = mag_thresh(gray, sobel_kernel=9, thresh=(30, 150))
    gradient_dir = dir_thresh(gray, sobel_kernel=9, thresh=(0.8, 1.2))
      
    combined_gradients =   2 * get_binary(threshold_image(combined_rs, thresh=thresh)) * 255
    combined_gradients += (gradient_dir / np.max(gradient_dir) * 255)
    combined_gradients += 2 * (gradient_mag/ np.max(gradient_mag) * 255) 
    combined_gradients = combined_gradients / np.max(combined_gradients) * 255
    gauss_blur = cv2.GaussianBlur(combined_gradients, (3, 3), 0)
    binary = get_binary(threshold_image(gauss_blur, thresh=thresh))
    return binary, combined_rs

def get_warped(image):
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

	
def search_lanes(binary_warped):
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return (left_fit, right_fit)   

# detect lanes without searching again
def detect_lanes(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return (ploty, left_fitx, right_fitx)

def calculate_curvature(ploty, left_fit, right_fit):
    quadratic_coeff = 3e-4
    leftx = np.array([320 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
    rightx = np.array([640 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
    y_eval = np.max(ploty)
    
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/960 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
  
    # Now our radius of curvature is in meters
    radius = (left_curverad + right_curverad) / 2

    # Calculate car position wrt to road
    center_lane = ( np.min(leftx)*xm_per_pix + np.max(rightx)*xm_per_pix) / 2
    center_image = (1280 * xm_per_pix) / 2
    car_position = center_image  - center_lane
    return (left_curverad, right_curverad, radius, car_position)

def map_lanes_image(warped, image, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

class Line(object):
    def __init__(self, Minv):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([False])
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        self.Minv = Minv
        
    def find_lane(self, warped):
        if self.detected:
            ploty, left_fitx, right_fitx = detect_lanes(warped, self.previous_fit[0], self.previous_fit[1])
            self.compare_lane_points(left_fitx, right_fitx)
        else:
            left_fit, right_fit = search_lanes(warped)
            if left_fit.any() and right_fit.any():
                self.detected = True
                self.previous_fit = np.array([left_fit, right_fit])
            ploty, left_fitx, right_fitx = detect_lanes(warped, left_fit, right_fit)
        # Save XY fit
        if left_fitx.any() and right_fitx.any():
            self.current_fit = np.array([left_fitx, right_fitx])
            self.recent_xfitted.append(np.array([left_fitx, right_fitx]))
        return ploty
    
    def compare_lane_points(self, left_fitx, right_fitx):
        recent = self.recent_xfitted[-1]
        diff_left = left_fitx - recent[0]
        diff_right = right_fitx - recent[1]
        
    def map_lanes(self, image, warped, ploty):
        _, _, radius, car_position = calculate_curvature(ploty, self.current_fit[0], self.current_fit[1])
        self.radius_of_curvature = radius
        self.line_base_pos = car_position
        result_img = map_lanes_image(warped, image, self.current_fit[0], self.current_fit[1], ploty, self.Minv)
        return result_img
    
    def process_image(self, image):
        undist = cv2.undistort(image, mtx, dist, None, mtx)
        _, rs_threshold = combine_grad_thresholds(undist)
        warped = get_warped(rs_threshold)
        ploty = self.find_lane(warped)
        result_image = self.map_lanes(image, warped, ploty)
        return result_image

