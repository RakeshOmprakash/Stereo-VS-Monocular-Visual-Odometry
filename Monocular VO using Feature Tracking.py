import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm


class VODO():
    def __init__(self, dire):
        self.K, self.P = self.get_calib(os.path.join(dire, 'calib.txt'))
        self.gt_poses = self.get_poses(os.path.join(dire,"poses.txt"))
        
        self.images = self.get_images(os.path.join(dire,"image_0"))
        self.sift = cv2.SIFT_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=10, key_size=10, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def get_calib(filepath):
       
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    def get_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    def get_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    def transformation(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        
        kp1, des1 = self.sift.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.sift.detectAndCompute(self.images[i], None)
        matches = self.flann.knnMatch(des1, des2, k=2)

        good = []
        
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        
        draw_params = dict(matchColor = -1, 
                 singlePointColor = None,
                 matchesMask = None, 
                 flags = 2)

        img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1],kp2, good ,None,**draw_params)
        cv2.imshow("image", img3)
        cv2.waitKey(200)

        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        R, t = self.decomp_essential_mat(E, q1, q2)

        transformation_matrix = self.transformation(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        
        def relative_z(R, t):
            T = self.transformation(R, t)
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)

            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = relative_z(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]


def main():
    dire = "F:\Workspace\Cv\Project\Datasets\KITTI\dataset\sequences/00/"  
    vo = VODO(dire)

    play_trip(vo.images) 

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    plt.figure()
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(dire) + ".html")


if __name__ == "__main__":
    main()
