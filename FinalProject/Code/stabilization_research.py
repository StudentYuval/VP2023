import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from funcs_and_structs import *

def open_writer(path: str, params: VideoParameters) -> cv2.VideoWriter:
    return cv2.VideoWriter(path, 
                           params.FOURCC,
                           params.FPS,
                           (params.Width, params.Height),
                           True)

def find_points_to_track2(frame: np.ndarray, frame_height: int = 1080, frame_width: int = 1920) -> np.ndarray:
    min_distance = round(0.05*min(frame_height, frame_width))
    block_size = round(0.01*min(frame_height, frame_width))
    p0 = cv2.goodFeaturesToTrack(frame, 100, 0.001, min_distance, blockSize=block_size)
    return p0

def find_points_to_track1(frame: np.ndarray, num_points:int =100, frame_height: int = 1080, frame_width: int = 1920) -> np.ndarray:
    corners = cv2.cornerHarris(frame, 9, 11, 0.02, borderType=cv2.BORDER_REPLICATE)
    corners = cv2.dilate(corners, None)
    corners = cv2.threshold(corners, 0.001*corners.max(), 255, cv2.THRESH_BINARY)[1]
    p0 = np.argwhere(corners.T == 255).astype(np.float32)
    p0 = p0.reshape(-1, 1, 2)
    return p0

input_path = '/home/egz01/repos/VP2023/FinalProject/Inputs/INPUT.avi'
output_path = '/home/egz01/repos/VP2023/FinalProject/Outputs/stabilized.avi'

vc = cv2.VideoCapture(input_path)
ret, prev_frame = vc.read()

if not ret:
    print('failed to read first frame')
    exit(1)

params = VideoParameters(vc)
vw = open_writer(output_path, params)

vw.write(prev_frame)

lk_params = {
                "winSize":(21,21),
                "maxLevel":5,
                "criteria":(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 3, 0.01)
}

warped_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
orig_features = find_points_to_track(warped_gray)
disp_features = None
for i in tqdm(range(params.NumFrames-1)):
    ret, cur_frame = vc.read()
    if not ret:
        print('failed to read next frame')
        break

    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    orig_features = find_points_to_track1(warped_gray)
    #disp_features = find_points_to_track1(cur_gray)
    disp_features, st, err = cv2.calcOpticalFlowPyrLK(warped_gray, cur_gray, orig_features, disp_features, **lk_params)
    distances = np.sqrt(np.power(orig_features - disp_features,2).sum(axis=2))
    mean_distance = distances[st == 1].mean()
    threshold = 10*mean_distance
    disp_points = disp_features[(st == 1) & (distances < threshold)]
    orig_points = orig_features[(st == 1) & (distances < threshold)]
    
    M, _ = cv2.estimateAffine2D(disp_points, orig_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    warped = cv2.warpAffine(cur_frame, M, (params.Width, params.Height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    '''frame = cur_frame.copy()
    for disp, orig in zip(disp_points, orig_points):
        disp_point = disp.ravel().astype(np.int32)
        orig_point = orig.ravel().astype(np.int32)
        frame = cv2.line(frame, disp_point, orig_point, (0, 255, 0), 2)
        frame = cv2.circle(frame, disp_point, 3, (0, 0, 255), -1)
        frame = cv2.circle(frame, orig_point, 3, (255, 0, 0), -1)
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    to_show = np.concatenate((frame, cv2.resize(warped, (0, 0), fx=0.5, fy=0.5)), axis=1)
    cv2.imshow('frame', to_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('terminated by user')
        break'''
    
    vw.write(warped)

vc.release()
vw.release()
cv2.destroyAllWindows()