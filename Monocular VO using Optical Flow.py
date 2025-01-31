import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join


image_path = 'F:\Workspace\Cv\Project\Datasets\KITTI\dataset\sequences/08/image_0/'
pose_path = "F:\Workspace\Cv\Project\Datasets\KITTI\dataset\sequences/08/08.txt"
k = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02],
     [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]], dtype=np.float32)

MinFeature = 3000

traj = np.ones((800, 800, 3), dtype=np.uint8)*255
x_loc = []
z_loc = []
curr_rot = None
curr_trans = None


def load_data(image_path):
    image_list = [image_path+f for f in listdir(image_path) if isfile(join(image_path, f))]
    image_list.sort()
    return image_list

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

def track(image_ref, image_cur, px_ref):
    lk_params = dict(winSize  = (21, 21), 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2

def scale(gt, frame_id):  
    x_prev = float(gt[0, frame_id-1])
    y_prev = float(gt[1, frame_id-1])
    z_prev = float(gt[2, frame_id-1])
    x = float(gt[0, frame_id])
    y = float(gt[1, frame_id])
    z = float(gt[2, frame_id])
    return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

def features(curr, prev, k):
    det = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

    kp1 = det.detect(curr)
    kp1 = np.array([x.pt for x in kp1], dtype=np.float32)

    kp1, kp2 = track(curr, prev, kp1)
    E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)
    kp1 = kp2
    return kp1, R, t

image_list = load_data(image_path)
gt = load_gt(pose_path)
curr = cv2.imread(image_list[0], 0)
prev = cv2.imread(image_list[1], 0)
kp1, cur_R, cur_t = features(curr, prev, k)

for i in range(len(image_list)):
    ## read the new frame from the image paths list ## 
    curr = cv2.imread(image_list[i], 0)
    ## track the feature movement from prev frame to current frame ## 
    kp1, kp2 = track(prev, curr, kp1)
    ## find the rotation and translation matrix ##
    E, mask = cv2.findEssentialMat(kp2, kp1, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, kp2, kp1, k)
    ## find the change of the feature location ## 
    change = np.mean(np.abs(kp2 - kp1))
    ## find the scale of the movemnt from the ground truth trajectory ## 
    scaled = scale(gt, i)
    if scaled > 2:
        scaled = 1
    ## check if the vehicle not moving by check the change value ## 
    if change > 5:
        ## accumulate the translation and rotation to find the X, Y, Z locations ## 
        cur_t = cur_t + scaled * cur_R.dot(t)
        cur_R = R.dot(cur_R)
    if(kp1.shape[0] < MinFeature):
        det = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        kp2 = det.detect(curr)
        kp2 = np.array([x.pt for x in kp2], dtype=np.float32)
    kp1 = kp2
    prev = curr
    if i > 2 :
        x = cur_t[0]
        y = cur_t[1]
        z = cur_t[2]
    else:
        x, y, z = 0.0, 0.0, 0.0
    x_loc.append(x)
    z_loc.append(z)

    raw_x, raw_y = int(x)+400, 590 - int(z)
    true_x, true_y = int(gt[0, i])+400, 590 - int(gt[2, i])
    cv2.circle(traj, (raw_x,raw_y), 1, (0,0,255), 1)
    cv2.circle(traj, (true_x,true_y), 1, (0,255,0), 2)
    cv2.imshow('Road facing camera', curr)
    cv2.imshow('Trajectory', traj)

    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
