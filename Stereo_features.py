import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join


image_pathL = 'F:\Workspace\Cv\Project\Datasets\KITTI\dataset\sequences/00/image_0/'
image_pathR = 'F:\Workspace\Cv\Project\Datasets\KITTI\dataset\sequences/00/image_1/'

pose_path = "F:\Workspace\Cv\Project\Datasets\KITTI\dataset\sequences/00/00.txt"
calib_l = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
     [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]], dtype=np.float32)
calib_r = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02],
                    [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
                    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]],dtype=np.float32)

global kl, rl, tl, kr, rr, tr

kl, rl, tl, _, _, _,_ = cv2.decomposeProjectionMatrix(calib_l)
kr, rr, tr, _, _, _,_ = cv2.decomposeProjectionMatrix(calib_r)

def load_data(image_pathL, image_pathR):
    image_listL = [image_pathL+f for f in listdir(image_pathL) if isfile(join(image_pathL, f))]
    image_listL.sort()

    image_listR = [image_pathR+k for k in listdir(image_pathR) if isfile(join(image_pathR, k))]
    image_listR.sort()
    return image_listL, image_listR

def load_gt(gt_path):
    f = open(gt_path,"r") 
    line = f.readlines()
    x = []
    y = []
    z = []
    for k in line:
        x.append(k.split(' ')[3])
        y.append(k.split(' ')[7])
        z.append(k.split(' ')[11])
    f.close()
    gt =  np.stack((x, y, z)).astype(np.float32)
    return gt

def extract_features(image):

    sift = cv2.SIFT_create(nfeatures = 2000)
    kp,des = sift.detectAndCompute(image,None)
    return kp, des

def depthH(disparity,kl,tl,tr):
    f = kl[0, 0]
    b = tl[1] - tr[1]
    disparity[disparity == 0] = 0.1
    disparity[disparity == -1] = 0.1

    depth_map = np.ones(disparity.shape, np.single)
    depth_map[:] = f * b / disparity[:]

    return depth_map

def depth(imgL,imgR):

    min_disparity = 0
    window_size = 6
    stereo = cv2.StereoBM_create(minDisparity = min_disparity, numDisparities = 16*6, blockSize = 15)
                                 #P1 = 8*3*window_size**2, 
                                 #P2 = 32*3*window_size**2
                                 #mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    disparity = stereo.compute(imgL,imgR).astype(np.float32)/16

    #disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              #beta=0, norm_type=cv2.NORM_MINMAX)
    #disparity_SGBM = np.uint8(disparity_SGBM)
    depth_map = depthH(disparity,kl,tl,tr)

    return  depth_map

#def visualize_matches(image1, kp1, image2, kp2, match):
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)

def extracted_features(imgL,imgR):
    kp_list_L = []
    des_list_L = []
    kp_list_R = []
    des_list_R = []
    
    kpl ,desl = extract_features(imgL)
    kp_list_L.append(kpl)
    des_list_L.append(desl)
    
    kpr ,desr = extract_features(imgR)
    kp_list_R.append(kpr)
    des_list_R.append(desr)

    return kp_list_L,kp_list_R, des_list_L,des_list_R

def fil_match_features(des1, des2,dist_threshold):

    filtered_match = []
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
    search_params = dict(checks = 50)
    #index_params, search_params
    matcher = cv2.FlannBasedMatcher(index_params, search_params)    
    match = matcher.knnMatch(des1, des2, k=2)
    
    for m ,n in match:
        if m.distance < dist_threshold * n.distance:
            filtered_match.append(m)
    
    return filtered_match

def estimate_motion(match, kp1, kp2, k, depth1=None):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    objectpoints = []
    
    
    for m in match:
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt
        
        s = depth1[int(v1), int(u1)]
        
        if s < 1000:
            p_c = np.linalg.inv(k) @ (s * np.array([u1, v1, 1]))
            
            image1_points.append([u1, v1])
            image2_points.append([u2, v2])
            objectpoints.append(p_c)
        
    objectpoints = np.vstack(objectpoints)
    imagepoints = np.array(image2_points)
    
    _, rvec, tvec, _ = cv2.solvePnPRansac(objectpoints, imagepoints, k, None)
    
    rmat, _ = cv2.Rodrigues(rvec)
    
    return rmat, tvec, image1_points, image2_points

image_list_l,image_list_r = load_data(image_pathL,image_pathR)

imagel = cv2.imread(image_list_l[0],0)
imager = cv2.imread(image_list_r[0],0)
gt = load_gt(pose_path)
dist_threshold = 0.6

traj = np.ones((800, 800, 3), dtype=np.uint8)*255
x_loc = []
z_loc = []

trajectory = np.zeros((3,1))
robot_pose = np.eye(4)

for i in range(len(image_list_l)):

    imagel = cv2.imread(image_list_l[i],0)
    imager = cv2.imread(image_list_r[i],0)

    kpl1,kpr1,desl1,desr1 = extracted_features(imagel,imager)
    kpl2,kpr2,desl2,desr2 = extracted_features(cv2.imread(image_list_l[i+1],0),cv2.imread(image_list_r[i+1],0))

    filtered_match = fil_match_features(desl1,desl2,dist_threshold)
    depth_map = depth(imagel,imager)
    rmat, tvec, image1_points, image2_points = estimate_motion(filtered_match, kpl1, kpl2, calib_l[:3,:3], depth_map)
            
    current_pose = np.eye(4)
    current_pose[0:3, 0:3] = rmat
    current_pose[0:3, 3] = tvec.T
    
    robot_pose[i + 1] = robot_pose[i] @ np.linalg.inv(current_pose)
    
    position = robot_pose[i + 1] @ np.array([0., 0., 0., 1.])
    
    trajectory[:, 1] = position[0:3]
    x = trajectory[0]
    y = trajectory[1]
    z = trajectory[2]

    raw_x, raw_y = int(x)+400, 590 - int(z)
    true_x, true_y = int(gt[0, i])+400, 590 - int(gt[2, i])
    cv2.circle(traj, (raw_x,raw_y), 1, (0,0,255), 1)
    cv2.circle(traj, (true_x,true_y), 1, (0,255,0), 2)
    cv2.imshow('Road facing camera', imagel)
    cv2.imshow('Trajectory', traj)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        


