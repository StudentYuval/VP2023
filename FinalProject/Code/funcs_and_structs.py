import os
import cv2
import numpy as np

class VideoParameters:
    def __init__(self, capture: cv2.VideoCapture):
        self.FPS = int(capture.get(cv2.CAP_PROP_FPS))
        self.Width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.Height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FOURCC = int(capture.get(cv2.CAP_PROP_FOURCC))
        self.NumFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __repr__(self):
        return "VideoParamters:\n" \
               f"\tFPS: {self.FPS}\n" \
               f"\tWidth: {self.Width}\n" \
               f"\tHeight: {self.Height}\n" \
               f"\tFOURCC: {self.FOURCC}\n" \
               f"\tNumFrames: {self.NumFrames}\n"

class ProjectPaths:
    def __init__(self, suffix: str='', input_dir: str = './Inputs', output_dir: str = './Outputs'):
        self.Input = os.path.join(input_dir, 'INPUT.avi')
        self.Background = os.path.join(input_dir, "background.jpg")
        self.Stabilized = os.path.join(output_dir, f"stabilized_{suffix}.avi")
        self.Extracted = os.path.join(output_dir, f"extracted_{suffix}.avi")
        self.Binary = os.path.join(output_dir, f"binary_{suffix}.avi")
        self.Alpha = os.path.join(output_dir, f"alpha_{suffix}.avi")
        self.Matted = os.path.join(output_dir, f"matted_{suffix}.avi")
        self.Output = os.path.join(output_dir, f"OUTPUT_{suffix}.avi")
        self.Timing = os.path.join(output_dir, "timing.json")
        self.Tracking = os.path.join(output_dir, "tracking.json")

    def __repr__(self):
        return "ProjectPaths:\n" \
               f"\tInput: {self.Input}\n" \
               f"\tBackground: {self.Background}\n" \
               f"\tStabilized: {self.Stabilized}\n" \
               f"\tExtracted: {self.Extracted}\n" \
               f"\tBinary: {self.Binary}\n" \
               f"\tAlpha: {self.Alpha}\n" \
               f"\tMatted: {self.Matted}\n" \
               f"\tOutput: {self.Output}\n" \
               f"\tTiming: {self.Timing}\n" \
               f"\tTracking: {self.Tracking}\n"
    
def open_writer(path: str, params: VideoParameters) -> cv2.VideoWriter:
    return cv2.VideoWriter(path, 
                               params.FOURCC,
                               params.FPS,
                               (params.Width, params.Height),
                               True)

def enhance_edges(frame: np.ndarray):
    blurred = cv2.GaussianBlur(src=frame, ksize=(25,25), sigmaX=4, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    #blurred = cv2.medianBlur(src=frame, ksize=3)
    #blurred = frame.copy()
    dx = cv2.Sobel(blurred, -1, 1, 0, ksize=5)
    dy = cv2.Sobel(blurred, -1, 0, 1, ksize=5)
    return cv2.addWeighted(np.abs(dx), 0.5, np.abs(dy), 0.5, 0)

def find_points_to_track(frame: np.ndarray, frame_height: int = 1080, frame_width: int = 1920,
                         mask:np.ndarray = None, max_num_points: int = 0) -> np.ndarray:
    
    min_distance = round(0.05*min(frame_height, frame_width))
    block_size = round(0.01*min(frame_height, frame_width))
    
    if mask is not None and np.count_nonzero(mask) < 0.5*mask.size:
        mask = None
    if max_num_points == 0:
        max_num_points = 100

    p0 = cv2.goodFeaturesToTrack(frame, max_num_points, 0.001, min_distance, blockSize=block_size, mask=mask)
    return p0

def calc_mse_between_frames(prev: np.ndarray, cur: np.ndarray) -> float:
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    mse = np.power(prev_gray - cur_gray, 2).mean()
    return mse
