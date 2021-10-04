#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import scipy.optimize as sciopt
import math
import cv2
import argparse
    
def forward_projection(p3D, cam): 
    '''
     Projects 3D points to 2D image points using the camera
     '''
    fx = cam['fx']
    fy = cam['fy']
    cx = cam['cx']
    cy = cam['cy']

    xs = fx*p3D[:,0]/p3D[:,2] + cx
    ys = fy*p3D[:,1]/p3D[:,2] + cy
    return xs, ys

def inverse_projection(img_pt, disparity, cam):
    ''' INVERSE_PROJECTION
    returns the world 3D coordinates from an image point and its stereo disparity
    '''
    fx = cam['fx']
    fy = cam['fy']
    cx = cam['cx']
    cy = cam['cy']
    baseline = cam['baseline']

    Z = fx*baseline/disparity
    X = (img_pt[0]-cx)*Z/fx
    Y = (img_pt[1]-cy)*Z/fy
    return X,Y,Z

def calcError(x1, z1, x2, z2):
    ''' get the root mean squared error between two x-z paths
    '''
    d = (x1-x2)**2 + (z1-z2)**2
    rms = np.sqrt(np.mean(d))
    return rms

def epipolar_match(matches, l_keypoints, r_keypoints, cam, y_thresh, d_thresh = 1):
    ''' given keypoint matches between left and right cameras, remove matches that:
        1) differ in y values by more than y_thresh (epipolar constraint says they should have same y)
        2) have small disparity < d_thresh 
        3) have large disparity > d_thresh + d_max
    '''
    fx = cam['fx']
    fy = cam['fy']
    baseline = cam['baseline']
    camera_height = cam['camera_height']
    d_max = 1.5*fx*baseline*cx/(fy*camera_height)
    # keep track of number of matches removed
    num_removed = 0
    for match in matches[::-1]: # work backwards in list so can remove items
        l_kp = l_keypoints[match.queryIdx].pt
        r_kp = r_keypoints[match.trainIdx].pt
        deltay = abs(r_kp[1] - l_kp[1])
        disparity = abs(r_kp[0] - l_kp[0])
        if disparity < d_thresh: # too small
            # remove this match
            num_removed += 1
            matches.remove(match)

        # finish this part
        elif deltay > y_thresh: # too far apart along y axis
            # remove this match
            num_removed += 1
            matches.remove(match)
        elif disparity > d_thresh + d_max:
            num_removed += 1
            matches.remove(match) # threshold on the disparity
    
    return num_removed
                   
def forward_motion_match(matches, kps1, kps2, cam, motion_thresh):
    ''' FORWARD MOTION MATCH - given keypoint matches between same camera one time step apart, remove matches that:
        1) are greater than expected motion distance (in pixels) which is product of motion_thresh and distance from center/fx
    '''
    cx = cam['cx']
    cy = cam['cy']
    center = np.array([cy, cx])
    # number of matches removed
    num_removed = 0

    
    for match in matches[::-1]:
        kp1 = np.array( kps1[match.queryIdx].pt )
        kp2 = np.array( kps2[match.trainIdx].pt )
        distance = math.sqrt(pow((kp1[0]-kp2[0]),2)+pow((kp1[1]-kp2[1]),2))
        distance_center = math.sqrt(pow(kp2[0]-cx,2)+pow(kp2[1]-cy,2))
        if distance > motion_thresh*distance_center:
            num_removed += 1
            matches.remove(match) 
            
         
    return num_removed

def determine_motion2D(image, pts_2D, pts_3D, cam, draw=False, out_thresh= 3.0):
    '''     DETERMINE MOTION 2D
    Given a 3D point cloud, Find the motion (rotation and translation) of the cloud
    that makes the re-projection of the 3D points match as closely as possible the 2D image points
    Outliers are considered points with residual error > mean + out_thresh*std of residuals
    '''
    image_x, image_y = pts_2D[:,0], pts_2D[:,1]
    #  put 3d points into homogeneous form by adding ones
    pts_3D_ho = np.insert(pts_3D, 3, 1, axis=1)
    
    # Apply Motion
    # Internal function that applies 3D rotation and translation to pts_3D array
    def apply_motion(motion):
        theta, d = motion
        # rotate and translate the 3D points
        c, s = np.cos(theta), np.sin(theta)
        tz = d*c
        tx = d*s
        RT = np.array([[ c, 0, s, tx],  #  Answer for number 5
                       [ 0, 1, 0,  0],
                       [-s, 0, c, tz],
                       [ 0, 0, 0,  1]])
        
        # transform the 3D points. 
        new_pts_3D = RT.dot(pts_3D_ho.T)
        # convert back to cartesian
        new_pts_3D = new_pts_3D[:3,:]
        return new_pts_3D.T
    
    # err_function
    # Calculates the reprojection error afer motion is applied to points
    def err_function(motion):
        new_pts_3D = apply_motion(motion)
        # calc new projections into image
        r_xs, r_ys = forward_projection(new_pts_3D, cam)
        # calc re-projection error
        residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
        ssderror = np.sqrt( np.sum( residuals ) )
        return ssderror
    
    # start of function
    r_xs, r_ys = forward_projection(pts_3D, cam)
    residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
    # print("initial reprojection error: ",np.sqrt( np.sum( residuals )))
    if draw:

        # draw the two sets of points
        #fig, ax = plt.subplots(figsize=(20,5))
        #plt.hold(True)
        plt.clf()
        plt.imshow(image)
        
        #ax.set_title('Optical Flow')
        plt.plot(r_xs,r_ys,'+',image_x,image_y,'o')
        for i in range(len(r_xs)):
            plt.plot([image_x[i], r_xs[i]],[image_y[i],r_ys[i]],'-')
        #plt.show()
        plt.pause(0.00001)
        #plt.imshow(image)
        
   
    outliers = True
    while outliers:
        # initial estimate is no motion:
        motion0 = [0, 0]  # equals rotation angle theta, and translation d
        
        # set bounds on possible rotation and motion between frames (no backwards allowed!)
        bounds = ( (-math.pi/18.0, math.pi/18.0 ), (0, 2000))
            
        res = sciopt.minimize(err_function, motion0, method = "SLSQP", bounds = bounds)
        motion = res.x
        # print(res.message)
        # print(" Minimization error: ", err_function(motion))
        
        # calculate residuals
        new_pts_3D = apply_motion(motion) 
        r_xs, r_ys = forward_projection(new_pts_3D, cam)
        residuals = (image_x-r_xs)**2 + (image_y-r_ys)**2
        mean, std = np.mean(residuals), np.std(residuals)
##        if draw:
##            plt.plot(residuals,'o')
##            plt.plot([0,len(residuals)], [mean, mean])
##            plt.plot([0,len(residuals)], [mean + out_thresh*std, mean + out_thresh*std])
##            plt.title("Residuals")
##            plt.show()
     
        #remove outliers and re-calculate motion
        o_threh = max(mean + out_thresh*std, 16.0) # assumes 4 pixel typical max error (16 = 4^2)
        remove = np.where(residuals>o_threh)
        # print("removing ", len(remove[0]), "outliers")
        if len(remove[0])>0:
            pts_3D_ho = np.delete(pts_3D_ho, remove, axis=0)
            image_x =np.delete(image_x, remove)
            image_y = np.delete(image_y, remove)
        else:
            outliers = False
    
    # draw the final two sets of points
##    if draw:
##        new_pts_3D = apply_motion(motion) 
##        r_xs, r_ys = forward_projection(new_pts_3D, cam)
##        fig, ax = plt.subplots(figsize=(20,5))
##        plt.imshow(image)
##        ax.set_title('Reprojection Error')
##        plt.plot(r_xs,r_ys,'+',image_x,image_y,'o')
##        for i in range(len(r_xs)):
##            plt.plot([image_x[i], r_xs[i]],[image_y[i],r_ys[i]],'-')
##        plt.show()
    print("Final Re-re projection error: ", err_function(motion))
    
    theta, d = motion
    c, s = np.cos(theta), np.sin(theta)
    tz = d*c
    tx = d*s
    RT = np.array([[ c, 0, s, tx/1000.], # translation in meters, not millimeters
                   [ 0, 1, 0,  0],
                   [-s, 0, c, tz/1000.],
                   [ 0, 0, 0,  1]])
    return RT # transformInverse(RT)

'''        
# MAIN CODE

'''

ap = argparse.ArgumentParser(
        description="Visual Odometry")

#ap.add_argument("-model", required=True, help="Model snapshot")
#ap.add_argument("-classes", required=False, help="Class CSV File")
ap.add_argument("-V", "-verbose", dest="verbose", action="store_true")
ap.add_argument("-d", action="store_true", help="Draw Images at each step")
ap.add_argument("-start", dest="start", type = int, required=False, help="start frame", default = 0)
ap.add_argument("-end", dest="end", type = int, required=False, help="end frame", default = 999)
args = ap.parse_args()

show_images = True
frame_start = 0
frame_end = 999

# Global Variables
speed_thresh = 10 # max number of pixels a point can move between time frames due to rotaton or translation alone 2
vertical_thresh = 1 # max number of pixels up or down allowed in epipolar matching (ideal epipolar has zero pixels up/down) 2
outlier_threshold = 1 # threshold on error residuals for determining outliers. outlier = mean + outlier_thresh*std 2
min_num_3D_points = 1 # min number of 3D stereo points between time frames needed to calculate motion 2
min_disparity = 1 # minimum stereo disparity to use

root_pathname = "/home/dhaval/Downloads/KITTIVO/KITTIVO/"
image_folder = "images/"
calib_folder = "calibs/"
poses_folder = "poses/"
sequence_number = "00/"
left_camera = "cam00/"
right_camera = "cam01/"

# read camera calibration file
calib_path = root_pathname+calib_folder+sequence_number
calib_file = "calib.txt"

# Load the camera Calibration information
try:
    with open(calib_path+calib_file, 'r') as f:
       line = f.readline()
       K_left = np.fromstring(line[4:], dtype=float, sep=' ')
       K_left = K_left.reshape(3,4)
       line = f.readline()
       K_right = np.fromstring(line[4:], dtype=float, sep=' ')
       K_right = K_right.reshape(3,4)

except FileNotFoundError:
    print('Unable to open calibration file: ' + calib_file)

# extract camera intrinsics from K matrices
fx, fy = K_left[0,0], K_left[1,1]
cx, cy = K_left[0,2], K_left[1,2]
camera_height = 1600 # 1.6 meters
baseline = 540 # 54cm. It should equal -K_right[0,3] but doesn't

# create a dict of camera information
camera_data = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline, "camera_height": camera_height}

# Get Poses for ground truth trajectory
# each pose is the transform from the car's system at time t back to the origin (time t=0)
# pose(t) = origin_T_car(t)
# so the translation component is the vector from the origin to the car
pose_path = root_pathname+poses_folder
pose_file = os.path.join(pose_path, sequence_number[:-1] + '.txt')
poses = []
try:
    with open(pose_file, 'r') as f:
       lines = f.readlines()

    for line in lines:
        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
        T_w_cam0 = T_w_cam0.reshape(3, 4)
        poses.append( T_w_cam0 )

except FileNotFoundError:
            print('Unable to open pose file: ' + pose_file)

# Create an array for storing our motion estimates for every pair of frames
# consisting of 4x4 transforms
motion_estimates = []

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Initiate ORB detector
orb = cv2.ORB_create()

# loop over images
frame = frame_start
while frame<=frame_end:
    frame_number = str(frame).zfill(6)
    print("Working on images ", frame_number," out of ", frame_end)
    
    # load the image into a NUMPY array
    left_img_file = root_pathname+image_folder+sequence_number+left_camera+frame_number+'.png'
    l_image = cv2.imread(left_img_file)
    right_img_file = root_pathname+image_folder+sequence_number+right_camera+frame_number+'.png'
    r_image = cv2.imread(right_img_file)

    # find the keypoints and descriptors 
    l_kps, l_des = orb.detectAndCompute(l_image,None)
    r_kps, r_des = orb.detectAndCompute(r_image,None)
    # each keypoint opject in list l_kps will have the 2D image point in .kp. For example l_kps[0].kp

    # Match descriptors between left and right camera
    current_lr_matches = bf.match(l_des, r_des)
    # current_lr_matches is a list of CVMatch objects. Each object has 'distance', 'queryIdx', 'trainIdx' properties
    # for each match in the list: l_kps[match.queryIdx] is matched with r_kps[match.trainIdx]
    # you are quaranteed that match.queryIdx < len(l_kp) and match.trainIdx < len(r_kp)

    # Sort them in the order of their distance.
    current_lr_matches = sorted(current_lr_matches, key = lambda x:x.distance)
    # print("Number of l-r matches: ",len(current_lr_matches))

##   ''' 
##    if show_images: # draw the matches
##        fig, ax = plt.subplots(figsize=(20,5))
##        match_image = l_image.copy()
##        match_image = cv2.drawMatches(l_image, l_kps, r_image, r_kps, current_lr_matches, match_image, flags=2)
##        plt.title('Raw Left-Right Correspondences')
##        plt.imshow(match_image),plt.show()
##    '''
    # Do epipolar constrained matching
    num_removed = epipolar_match(current_lr_matches, l_kps, r_kps, camera_data, vertical_thresh, d_thresh = min_disparity)
    # print("Number of epipolar matches removed: ", num_removed)
##    '''
##    if show_images: # draw the matches
##        fig, ax = plt.subplots(figsize=(20,5))
##        match_image = l_image.copy()
##        match_image = cv2.drawMatches(l_image, l_kps, r_image, r_kps, current_lr_matches, match_image, flags=2)
##        plt.title('Filtered Left-Right Correspondences')
##        plt.imshow(match_image),plt.show()
##    '''
    # now match features with previous frame
    if frame > frame_start:
        # Match descriptors between previous left and current left camera
        previous_l_current_l_matches = bf.match(previous_l_des, l_des)
        num_removed = forward_motion_match(previous_l_current_l_matches, previous_l_kps, l_kps, camera_data, speed_thresh)
        # print("Number of previous motion matches removed: ",num_removed)
        '''
        if show_images:
            fig, ax = plt.subplots(figsize=(20,5))
            match_image = l_image.copy()
            match_image = cv2.drawMatches(previous_l_img, previous_l_kps, l_image, l_kps, previous_l_current_l_matches, match_image, flags=2)
            plt.title('Left to Left Temporal Correspondences')
            plt.imshow(match_image), plt.show()
        '''
      
        # for every point in previous frame that we matched for stereo
        # see if we also matched to current frame in time 
        overlap = list(set([m.queryIdx for m in previous_lr_matches]) & set([m.queryIdx for m in previous_l_current_l_matches]))
        # print("Found ",len(overlap)," previous left stereo matches that mapped to new left frame")
        
        # now get 3D points from stereo match for those points with matches (in overlap)  
        num_p1 = 0
        time_matches = []
        points1_3D, points2_3D = [], []
        points1_2D = []
        # for every stereo match in overlap get 3D points from stereo
        for m1 in previous_l_current_l_matches:
            if m1.queryIdx in overlap:
                disparity1 = None
                for m2 in previous_lr_matches:
                    if m2.queryIdx == m1.queryIdx:
                        l_img_point1 = previous_l_kps[m2.queryIdx].pt
                        r_img_point1 = previous_r_kps[m2.trainIdx].pt
                        disparity1 = l_img_point1[0] - r_img_point1[0]
                # find match in current time frame
                disparity2 = None
                for m2 in current_lr_matches:
                    if m2.queryIdx == m1.trainIdx:
                        l_img_point2 = l_kps[m2.queryIdx].pt
                        r_img_point2 = r_kps[m2.trainIdx].pt
                        disparity2 = l_img_point2[0] - r_img_point2[0]
                
                if disparity2 is not None and disparity1 is not None: # None if we couldn't find match in previous-current left matches to previous time left-right matches"
                    # get the two pairs of 3D points from disparities
                    X1,Y1,Z1 = inverse_projection(l_img_point1, disparity1, camera_data)
                    X2,Y2,Z2 = inverse_projection(l_img_point2, disparity2, camera_data)
                    points1_2D.append(l_img_point1)                    
                    points1_3D.append([X1,Y1,Z1])
                    points2_3D.append([X2,Y2,Z2])
                    num_p1 += 1
        
        # create lists of x and y locations in the current frame
        # for the points that have stereo matches and temporal matches
        if num_p1 > min_num_3D_points:
            points1_3D = np.array(points1_3D)
            points2_3D = np.array(points2_3D)
            points1_2D = np.array(points1_2D)
        
            transform = determine_motion2D(l_image, points1_2D, points2_3D, camera_data, draw=show_images, out_thresh=outlier_threshold)
            # print("Estimated transform {} from {} points.".format( transform, num_p1))
            motion_estimates.append(transform)
        else:
            print("Not enough matches to calculate motion!")
            # repeat the previous motion as our best guess
            motion_estimates.append(motion_estimates[-1])
        
    # copy current stuff to previous stuff
    previous_r_img = r_image
    previous_r_kps = r_kps
    previous_r_des = r_des
    
    previous_l_img = l_image
    previous_l_kps = l_kps
    previous_l_des = l_des
    
    previous_lr_matches = current_lr_matches
    
    frame = frame+1
# END Frame loop

# extract the x and z coordinates from the ground truth poses 
x_pos_truth = []
z_pos_truth = []
# extract X and Z values
for i in range(frame_end-frame_start+1):
    x_pos_truth.append(poses[i+frame_start][0,3])
    z_pos_truth.append(poses[i+frame_start][2,3])

# estimate path of car and compare to ground truth
# each motion_estimate is the transform from the car's position at time t-1 to time t
# motion_estimate(t) = car(t-1)_T_car(t)
# we get origin_T_car(t) by contatenating origin_T_car(t0) car(t0)_T_car(t0+1) cart(t0+1)_T_car(t0+2) ... car(t-1)_T_car(t)
first_pose = np.eye(4)
first_pose[:3,:] = poses[frame_start]
x_pos_est = [first_pose[0,3]]
z_pos_est = [first_pose[2,3]]
pose_estimates = [first_pose]
for t in motion_estimates:
    new_pose = np.dot(pose_estimates[-1], t)
    pose_estimates.append(new_pose)
    x_pos_est.append(new_pose[0,3])
    z_pos_est.append(new_pose[2,3])

# calculate the RMS error over the path
x_pos_est = np.array(x_pos_est)
z_pos_est = np.array(z_pos_est)
x_pos_truth = np.array(x_pos_truth)
z_pos_truth = np.array(z_pos_truth)
rms = calcError(x_pos_est, z_pos_est, x_pos_truth, z_pos_truth)
print("RMS error over the X-Zpath: ", rms)
plt.figure()
# plot the paths and score
plt.plot(x_pos_est, z_pos_est)
plt.plot(x_pos_est, z_pos_est,'o')
plt.plot(x_pos_truth, z_pos_truth, '+')
s = "RMS: {rms}"
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 14,
        }
xt = np.min(x_pos_est)
yt = np.min(z_pos_est)
plt.text(xt, yt, s, fontdict = font)

plt.title('Estimated (o) vs Ground Truth (+) Motion')
plt.show()
