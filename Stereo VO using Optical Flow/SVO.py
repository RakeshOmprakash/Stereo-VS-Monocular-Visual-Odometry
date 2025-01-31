import numpy as np
import os
from scipy.optimize import least_squares
import cv2

from graph.plotting import plot

from tqdm import tqdm

class VOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self.loadcalib(data_dir + '/calib.txt')
        self.groundtruth_poses = self.loadposes(data_dir + '/poses.txt')
        self.img_l = self.loadimages(data_dir + '/image_0')
        self.img_r = self.loadimages(data_dir + '/image_1')

        blk = 11
        P1 = blk * blk * 8
        P2 = blk * blk * 32
        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=blk, P1=P1, P2=P2)
        self.disparities = [np.divide(self.disparity.compute(self.img_l[0], self.img_r[0]).astype(np.float32), 16)]
        self.FF = cv2.FastFeatureDetector_create()

        self.lkparams = dict(winSize=(15, 15), flags=cv2.MOTION_AFFINE, maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

    @staticmethod
    def loadimages(filepath):

        imgpaths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        imgs = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in imgpaths]
        return imgs
    
    @staticmethod
    def loadposes(filepath):
        po = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = np.fromstring(line, dtype=np.float64, sep=' ')
                values = values.reshape(3, 4)
                values = np.vstack((values, [0, 0, 0, 1]))
                po.append(values)
        return po

    @staticmethod
    def loadcalib(filepath):
        with open(filepath, 'r') as f:
            par = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(par, (3, 4))
            K_l = P_l[0:3, 0:3]
            par = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(par, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def formtransf(R, t):
        value = np.eye(4, dtype=np.float64)
        value[:3, :3] = R
        value[:3, 3] = t
        return value

    def reprojectionresiduals(self, dof, q1, q2, Q1, Q2):

        rotation = dof[:3]
        Rmatrix, _ = cv2.Rodrigues(rotation)
        translation = dof[3:]
        transf = self.formtransf(Rmatrix, translation)

        fprojection = np.matmul(self.P_l, transf)
        bprojection = np.matmul(self.P_l, np.linalg.inv(transf))

        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        q1pred = Q2.dot(fprojection.T)
        q1pred = q1pred[:, :2].T / q1pred[:, 2]

        q2pred = Q1.dot(bprojection.T)
        q2pred = q2pred[:, :2].T / q2pred[:, 2]

        errors = np.vstack([q1pred - q1.T, q2pred - q2.T]).flatten()
        return errors

    def tilekeypoints(self, img, tile_h, tile_w):

        def findkeypoints(x, y):
            
            section = img[y:y + tile_h, x:x + tile_w]
            keypts = self.FF.detect(section)
            for pt in keypts:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            if len(keypts) > 10:
                keypts = sorted(keypts, key=lambda x: -x.response)
                return keypts[:10]
            return keypts
        h, w, *_ = img.shape

        keypointslist = [findkeypoints(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        keypointslistflatten = np.concatenate(keypointslist)
        return keypointslistflatten

    def trackkeypoints(self, img1, img2, kp1, max_error=4):

        trackpts1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        trackpts2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpts1, None, **self.lkparams)

        track = st.astype(bool)

        belowthreshold = np.where(err[track] < max_error, True, False)

        trackpts1 = trackpts1[track][belowthreshold]
        trackpts2 = np.around(trackpts2[track][belowthreshold])

        h, w = img1.shape
        withinbounds = np.where(np.logical_and(trackpts2[:, 1] < h, trackpts2[:, 0] < w), True, False)
        trackpts1 = trackpts1[withinbounds]
        trackpts2 = trackpts2[withinbounds]

        return trackpts1, trackpts2

    def calculaterightkeypoints(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):

        def findindex(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        
        disp1, mask1 = findindex(q1, disp1)
        disp2, mask2 = findindex(q2, disp2)
        
        withinbounds = np.logical_and(mask1, mask2)
        
        q1_l, q2_l, disp1, disp2 = q1[withinbounds], q2[withinbounds], disp1[withinbounds], disp2[withinbounds]
        
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        
        return q1_l, q1_r, q2_l, q2_r

    def calc3dpoints(self, q1_l, q1_r, q2_l, q2_r):

        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        Q1 = np.transpose(Q1[:3] / Q1[3])

        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def calculatepose(self, q1, q2, Q1, Q2, max_iter=100):

        threshold = 5

        minimum_error = float('inf')
        thres = 0
        for _ in range(max_iter):
            randomindex = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[randomindex], q2[randomindex], Q1[randomindex], Q2[randomindex]
            in_guess = np.zeros(6)
            opt_res = least_squares(self.reprojectionresiduals, in_guess, method='lm', max_nfev=200, args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            err = self.reprojectionresiduals(opt_res.x, q1, q2, Q1, Q2)
            err = err.reshape((Q1.shape[0] * 2, 2))
            err = np.sum(np.linalg.norm(err, axis=1))

            if err < minimum_error:
                minimum_error = err
                out_pose = opt_res.x
                thres = 0
            else:
                thres += 1
            if thres == threshold:
                break

        rotation = out_pose[:3]
        Rmatrix, _ = cv2.Rodrigues(rotation)
        translation = out_pose[3:]
        transformation_matrix = self.formtransf(Rmatrix, translation)
        return transformation_matrix

    def getpose(self, i):

        img1_l, img2_l = self.img_l[i - 1:i + 1]

        kp1_l = self.tilekeypoints(img1_l, 10, 20)

        tp1_l, tp2_l = self.trackkeypoints(img1_l, img2_l, kp1_l)

        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.img_r[i]).astype(np.float32), 16))

        tp1_l, tp1_r, tp2_l, tp2_r = self.calculaterightkeypoints(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        Q1, Q2 = self.calc3dpoints(tp1_l, tp1_r, tp2_l, tp2_r)

        transformation_matrix = self.calculatepose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix


def main():
    #Change the path directory
    data_dir = '/Users/rakesh/Documents/Visual Studio Code/KITTI_sequence_2' 
    vo = VOdometry(data_dir)
    
    groundtruth_path = []
    calculated_path = []
    for i, groundtruth_pose in enumerate(tqdm(vo.groundtruth_poses, unit="poses")):
        if i < 1:
            current_pose = groundtruth_pose
        else:
            transfmatrix = vo.getpose(i)
            current_pose = np.matmul(current_pose, transfmatrix)
        groundtruth_path.append((groundtruth_pose[0, 3], groundtruth_pose[2, 3]))
        calculated_path.append((current_pose[0, 3], current_pose[2, 3]))
    plot.visualize_paths(groundtruth_path, calculated_path, "Graph of SVO", file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()