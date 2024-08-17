import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class Dataset_Handler():
    def __init__(self, sequence, lidar=True, progress_bar = True, low_memory = False):
        self.lidar = lidar
        self.low_memory = low_memory
        
        self.seq_dir = 'dataset/sequences/{}/'. format(sequence)
        self.poses_dir = 'dataset/poses/{}.txt'. format(sequence)
        
        self.left_image = os.listdir(self.seq_dir + 'image_0')
        self.right_image = os.listdir(self.seq_dir + 'image_1')
        self.lidar_files = os.listdir(self.seq_dir + 'velodyne')
        
        self.left_image.sort()
        self.right_image.sort()
        self.lidar_files.sort()
        
        self.num_frames = len(self.left_image)
        self.lidar_path = self.seq_dir + 'velodyne/'
        
        poses = pd.read_csv(self.poses_dir,delimiter=' ', header=None)
        self.gt = np.zeros((self.num_frames, 3, 4))
        
        for i in range(len(poses)):
            self.gt[i] = np.reshape(poses.iloc[i].values, (3,4))
            
        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape(3,4)
        self.P1 = np.array(calib.loc['P1:']).reshape(3,4)
        self.P2 = np.array(calib.loc['P2:']).reshape(3,4)
        self.P3 = np.array(calib.loc['P3:']).reshape(3,4)
        self.Tr = np.array(calib.loc['Tr:']).reshape(3,4)
        
        
        if low_memory:
            self.reset_frames()
            self.first_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image[0], 0)
            self.first_image_right = cv2.imread(self.seq_dir + 'image_1/' + self.right_image[0], 0)
            self.second_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image[1], 0)
            if lidar:
                self.first_pointcloud = np.fromfile(self.lidar_path + self.lidar_files[0], dtype=np.float32, count = -1).reshape(-1,4)
            self.imheight = self.first_image_left.shape[0]
            self.imwidth = self.first_image_left.shape[1]
        else:
            self.image_left = []
            self.image_right = []
            self.pointclouds = []
            for i, name_left in enumerate(self.left_image):
                name_right = self.right_image[i]
                self.image_left.append(cv2.imread(self.seq_dir + 'image_0/' + name_left, 0))
                self.image_right.append(cv2.imread(self.seq_dir + 'image_1/' + name_right, 0))
                if lidar:
                    point_cloud = np.fromfile(self.lidar_path + self.lidar_files[i], dtype=np.float32).reshape(-1,4)
                    self.pointclouds.append(point_cloud)
                self.imheight = self.image_left[0].shape[0]
                self.imwidth = self.image_left[0].shape[1]
    
    def reset_frames(self):
        self.image_left = (cv2.imread(self.seq_dir + 'image_0/' + name_left, 0) for name_left in self.left_image)
        self.image_right = (cv2.imread(self.seq_dir + 'image_1/' + name_right, 0) for name_right in self.right_image)
        if self.lidar:
            self.pointclouds = (np.fromfile(self.lidar_path + name, dtype=np.float32, count=-1).reshape(-1,4) for name in self.lidar_files) 
        pass
    

def compute_left_disparity_map(img_left, img_right, matchers='bm', rgb=False, verbose=False):
    sad_window = 6
    num_disparities = sad_window*16
    block_size = 11
    matchers_name = matchers
    
    if matchers_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    elif matchers_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities, 
                                        minDisparity = 0, 
                                        blockSize=block_size, 
                                        P1 = 8*1*block_size**2, 
                                        P2 = 32*1*block_size**2, 
                                        disp12MaxDiff = 1,
                                        uniquenessRatio = 10,
                                        speckleWindowSize = 100,
                                        speckleRange = 32,
                                        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    
    if rgb:
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    
    start = datetime.datetime.now()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = datetime.datetime.now()
    
    if verbose:
        print('Time to comp disparity', end-start)
    
    return disp_left

def decompose_projection_matrix(P):
    k,r,t,_,_,_,_ = cv2.decomposeProjectionMatrix(P)
    t = (t/t[3])[:3]
    return k,r,t

def calc_depth_map(disp_left, k_left, t_left, t_right, rectified = "True"):
    if rectified:
        baseline = t_right[0] - t_left[0]
    else:
        baseline = t_left[0] - t_right[0]
    
    f = k_left[0][0]
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1
    
    depth_map = np.ones(disp_left.shape)
    depth_map = f*baseline/disp_left
    
    return depth_map

def stereo_2_depth(img_left, img_right, P0, P1, matchers='bm', rgb=False, verbose=True, 
                   rectified=True):
    # Compute disparity map
    disp = compute_left_disparity_map(img_left, img_right, matchers, rgb, verbose)
    
    #decompose projection matrix
    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)
    
    # calculate depth map
    depth_map = calc_depth_map(disp, k_left, t_left, t_right, rectified)
    
    return depth_map


def pointcloud2image(pointcloud, imheight, imwidth, Tr, P0):
    pointcloud = pointcloud[pointcloud[:,0] > 0]
    reflectance = pointcloud[:,3]
    
    #homogenous coordinates
    pointcloud = np.hstack([pointcloud[:,:3], np.ones(pointcloud.shape[0]).reshape((-1,1))])
    
    #transform pointcloud to camera coordinates
    cam_xyz = Tr.dot(pointcloud.T)
    
    #clip off neg z values
    cam_xyz = cam_xyz[: , cam_xyz[2] > 0]
    
    depth = cam_xyz[2].copy()
    
    
    cam_xyz /= cam_xyz[2]
    
    cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])
    projection = P0.dot(cam_xyz)
    
    pixel_coordinates = np.round(projection.T, 0)[:,:2].astype('int')
    
    indices = np.where( (pixel_coordinates[:,0] >= 0) & 
                        (pixel_coordinates[:,0] < imwidth) & 
                        (pixel_coordinates[:,1] >= 0) &
                        (pixel_coordinates[:,1] < imheight))
    
    pixel_coordinates = pixel_coordinates[indices]
    depth = depth[indices]
    reflectance = reflectance[indices]
    
    render = np.zeros((imheight, imwidth)) 
    for j, (u,v) in enumerate(pixel_coordinates):
        if u >= imwidth or u <0:
            continue
        if v >= imheight or v <0:
            continue
        
        render[v,u] = depth[j]
        

        
    return render

def extract_features(img, detector='sift', mask=None):
    if detector == 'sift':
        det = cv2.SIFT_create()
    if detector == 'orb':
        det = cv2.ORB_create(nfeatures=2500)
    
    kp, des = det.detectAndCompute(img, mask)
    
    return kp, des


def match_features(des1, des2, matcher='bf', detector='sift', sort=True, k=2, knn=True):
    if matcher == 'bf':
        if detector == 'sift':
            match = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        if detector == 'orb':
            match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    elif matcher == 'flann':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        match = cv2.FlannBasedMatcher(index_params, search_params)
    
    if knn:
        matches = match.knnMatch(des1, des2, k=k)
    else:
        matches = match.match(des1, des2)
    
    if sort:
        matches = sorted(matches, key = lambda x:x[0].distance)
        
    return matches

def visualize_matches(image1, kp1, image2, kp2, matches):
    img_matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches, None, flags=2)
    plt.figure(figsize=(16,10), dpi=100)
    plt.imshow(img_matches)
    plt.show()


def filter_matches(matches, dist_threshold):
    good_matches = []
    for m,n in matches:
        if m.distance <= dist_threshold*n.distance:
            good_matches.append([m])
    return good_matches

# im1 = cv2.imread('dataset/sequences/03/image_0/000390.png', 0)
# im2 = cv2.imread('dataset/sequences/03/image_0/000391.png', 0)
# kp1, des1 = extract_features(im1, detector='orb')
# kp2, des2 = extract_features(im2, detector='orb')
# matches = match_features(des1, des2, matcher='bf', detector='orb', sort=True)
# matches = filter_matches(matches, 0.45)
# visualize_matches(im1, kp1, im2, kp2, matches)
# exit()

def estimate_motion(matches, kp1, kp2, k, depth1, max_depth=1500):
    
    rmat = np.eye(3)
    tvec = np.zeros((3,1))
    
    im1_pts = []  # List to store keypoints from the first image
    im2_pts = []  # List to store keypoints from the second image

    for match_list in matches:
        for match in match_list:
            im1_pts.append(kp1[match.queryIdx].pt)  # Get keypoint from the first image
            im2_pts.append(kp2[match.trainIdx].pt)  # Get keypoint from the second image

    im1_pts = np.float32(im1_pts).reshape(-1, 1, 2)
    im2_pts = np.float32(im2_pts).reshape(-1, 1, 2)
    
    cx = k[0,2]
    cy = k[1,2]
    fx = k[0,0]
    fy = k[1,1]
    
    objects = np.zeros((0,3))
    
    delete = []
    
    for i, point in enumerate(im1_pts):
        u, v = point[0]
        z = depth1[int(round(v)), int(round(u))]
        
        if z > max_depth:
            delete.append(i)
            continue
        
        x = (u - cx)*z/fx
        y = (v - cy)*z/fy
        
        objects = np.vstack([objects, np.array([x,y,z])])
    
    print(len(im1_pts))
    print(len(delete))
    im1_pts = np.delete(im1_pts, delete, axis=0)
    im2_pts = np.delete(im2_pts, delete, axis=0)
    print("testtt", len(im1_pts))
    
    _, rvec, tvec, inliers = cv2.solvePnPRansac(objects, im2_pts, k, None)
    rmat = cv2.Rodrigues(rvec)[0]
    
    return rmat, tvec, im1_pts, im2_pts

def visual_odometry(handler, detector='sift', matcher='bf', filter_match_distance=None, stereo_matcher='sgbm', mask=None, subset=None, plot=False):
    
    #determine if handler has lidar data
    lidar = handler.lidar
    
    # Methods being used
    print('Generating disparities with stereo{}'.format(str.upper(stereo_matcher)))
    print('Detetcting features with {} and matching with {}'.format(str.upper(detector), matcher))
    
    if filter_match_distance is not None:
        print('Filtering matches with distance ratio:', filter_match_distance)
    
    if lidar:
        print('Extracting pointclouds from lidar data')
    
    if subset is not None:
        print('Processing subset of frames:', subset)
        num_frames = subset
    else:
        num_frames = handler.num_frames
        
    if plot:
        fig = plt.figure(figsize=(14,14))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = handler.gt[:,0,3]
        ys = handler.gt[:,1,3]
        zs = handler.gt[:,2,3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs,ys,zs, label='Ground Truth', c='k')
        
    #Homogenous coordinates
    
    T_tot = np.eye(4)
    traj = np.zeros((num_frames, 3, 4))
    traj[0] = T_tot[:3, :]
    imheight = handler.imheight
    imwidth = handler.imwidth
    
    #decompose
    k_left, r_left, t_left, _, _, _, _ = cv2.decomposeProjectionMatrix(handler.P0)
    
    if handler.low_memory:
        handler.reset_frames()
        img_plus1 = next(handler.image_left)
        
    #interation
    time = datetime.timedelta(0)
    for i in range(num_frames-1):
        start = datetime.datetime.now()
        
        if handler.low_memory:
            image_left = img_plus1
            img_plus1 = next(handler.image_left)
            image_right = next(handler.image_right)
        else:
            image_left = handler.image_left[i]
            image_right = handler.image_right[i]
            img_plus1 = handler.image_left[i+1]
        
        
        depth = stereo_2_depth(image_left, 
                               image_right, 
                               handler.P0, 
                               handler.P1, 
                               stereo_matcher, 
                               rgb=False, 
                               verbose=False)
        if lidar:
            if handler.low_memory:
                pointcloud = next(handler.pointclouds)
            else:
                pointcloud = handler.pointclouds[i]
            
            lidar_depth = pointcloud2image(pointcloud, 
                                           imheight, 
                                           imwidth, 
                                           handler.Tr, 
                                           handler.P0)
            
            # indices = np.where(lidar_depth < 3000)
            indices = np.where(lidar_depth > 0)
            
            depth[indices] = lidar_depth[indices]
            
        #ger kp and des
        kp0, des0 = extract_features(image_left, detector, mask)
        kp1, des1 = extract_features(img_plus1, detector, mask)
        
        #match features
        matches_unflit = match_features(des0, 
                                        des1, 
                                        matcher, 
                                        detector, sort=True)
        
        if filter_match_distance is not None:
            matches = filter_matches(matches_unflit, filter_match_distance)
        else:
            matches = matches_unflit
            
        
        #estimate motion
        rmat, tvec, im1_pts, im2_pts = estimate_motion(matches, 
                                                       kp0, 
                                                       kp1, 
                                                       k_left, 
                                                       depth)
        
        # Create a blank tranformation matrix
        
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot.dot(np.linalg.inv(Tmat))
        traj[i+1, :, :] = T_tot[:3, :]
        
        end = datetime.datetime.now()
        
        if plot:
            xs = traj[:i+2, 0, 3]
            ys = traj[:i+2, 1, 3]
            zs = traj[:i+2, 2, 3]
            #plot the match features at every step
            # cv2.imshow('matches', cv2.drawMatchesKnn(image_left, kp0, img_plus1, kp1, matches, None, flags=2))
            #use cv2 to plot the x,y,z coordinates
            # cv2.waitKey(1)
            plt.plot(xs, ys, zs, c='chartreuse')
            plt.pause(1e-32)
            
    if plot:
        plt.plot(traj[:,0,3], traj[:,1,3], traj[:,2,3], label='Estimated', c='chartreuse')
        plt.savefig('trajectory_orb_test1.png')
        # cv2.destroyAllWindows()
        # plt.close()
    
    print('Total time to compute:', time)
    print('Average time per frame:', time/num_frames)
    
    return traj

def calculate_error(ground_truth, estimated, error_type='mse'):
    nframes_est = estimated.shape[0]
    
    def get_mse(ground_truth, estimated):
        se = np.sqrt((ground_truth[nframes_est, 0, 3] - estimated[:, 0, 3])**2 
                    + (ground_truth[nframes_est, 1, 3] - estimated[:, 1, 3])**2 
                    + (ground_truth[nframes_est, 2, 3] - estimated[:, 2, 3])**2)**2
        mse = se.mean()
        return mse
    
    def get_mae(ground_truth, estimated):
        ae = np.sqrt((ground_truth[nframes_est, 0, 3] - estimated[:, 0, 3])**2 
                    + (ground_truth[nframes_est, 1, 3] - estimated[:, 1, 3])**2 
                    + (ground_truth[nframes_est, 2, 3] - estimated[:, 2, 3])**2)
        mae = ae.mean()
        return mae
    
    if error_type == 'mae':
        return get_mae(ground_truth, estimated)
    elif error_type == 'mse':
        return get_mse(ground_truth, estimated)
    elif error_type == 'rmse':
        return np.sqrt(get_mse(ground_truth, estimated))
    elif error_type == 'all':
        mae = get_mae(ground_truth, estimated)
        mse = get_mse(ground_truth, estimated)
        rmse = np.sqrt(mse)
        return {'mae': mae,
                'rmse': rmse,
                'mse': mse}

handler = Dataset_Handler('01', progress_bar=False, low_memory=True)
disp = compute_left_disparity_map(handler.first_image_left, handler.first_image_right, 'sgbm')
k_left, r_left, t_left = decompose_projection_matrix(handler.P0)
k_right, r_right, t_right = decompose_projection_matrix(handler.P1)
depth_map = calc_depth_map(disp, k_left, t_left, t_right)
mask = np.zeros((handler.imheight, handler.imwidth))
ymax = depth_map.shape[0]
xmax = depth_map.shape[1]
cv2.rectangle(mask, (96, 0), (xmax, ymax), (255), thickness = -1)



trajectory_test = visual_odometry(
    handler, 
    detector='orb', 
    matcher='bf', 
    filter_match_distance=0.55, 
    stereo_matcher='sgbm', 
    mask=mask, 
    subset=None, 
    plot=True
)