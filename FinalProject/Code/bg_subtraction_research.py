import cv2
import time
import numpy as np
from tqdm import tqdm
from funcs_and_structs import *
import matplotlib.pyplot as plt

def bg_subtraction_block(stabilized_path: str, 
                         extracted_path: str, binary_path: str, show_work=False) -> float:
    print("Extracting background...")

    start_time = time.time()

    vc = cv2.VideoCapture(stabilized_path)
    if not vc.isOpened():
        print('failed to open video')
        exit(1)

    params = VideoParameters(vc)
    vw_extracted = open_writer(extracted_path, params)
    vw_binary = open_writer(binary_path, params)

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=50, detectShadows=False)

    for i in tqdm(range(params.NumFrames)):
        ret, frame = vc.read()
        if not ret:
            print('failed to read next frame')
            break

        fg_mask = bg_subtractor.apply(frame)
        fg_mask[fg_mask < 254] = 0
        fg_mask[fg_mask >= 254] = 255

        vw_binary.write(fg_mask)
        frame[fg_mask == 0] = 0
        vw_extracted.write(frame)

        if show_work:
            cv2.imshow('frame', frame)
            cv2.imshow('fg_mask', fg_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vc.release()
    vw_extracted.release()
    vw_binary.release()
    cv2.destroyAllWindows()

    return time.time() - start_time

def set_roi(frame: np.ndarray, 
            y_offset_start: int, 
            y_offset_end: int,
            x_offset_start: int, 
            x_offset_end: int, ) -> np.ndarray:
    if len(frame.shape) == 2:
        val_to_set = 0
    else:
        val_to_set = [0,0,0]

    frame[:y_offset_start, :] = val_to_set
    frame[-y_offset_end:, :] = val_to_set
    frame[:, :x_offset_start] = val_to_set
    frame[:, -x_offset_end:] = val_to_set
    return frame

def train_bg_subtractor(vc: cv2.VideoCapture, 
                             offset_y: int=0,
                             offset_x: int=0,
                             blur_kernel_size: tuple[int, int]=(5,5),
                             learningRate: float=0.9,
                             history: int=30,
                             varThreshold: int=50, 
                             detectShadows: bool=False) -> tuple[cv2.BackgroundSubtractorMOG2, np.ndarray]:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
    #bg_subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=varThreshold, detectShadows=detectShadows)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in tqdm(range(num_frames)):
        ret, frame = vc.read()
        if not ret:
            print('failed to read next frame')
            exit(1)
        # frame = cv2.GaussianBlur(frame, blur_kernel_size, sigmaX=0, sigmaY=0)
        set_roi(frame, offset_y, offset_y, offset_x, offset_x)
        _ = bg_subtractor.apply(frame, learningRate=learningRate)

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return bg_subtractor, bg_subtractor.getBackgroundImage()

def apply_subtractor_to_video(vc: cv2.VideoCapture, 
                              bg_subtractor: cv2.BackgroundSubtractorMOG2,
                              learningRate=0.001,
                              offset_y: int=30, offset_x: int=100):
    
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

    Width = 400
    Height = 250
    window_size = (Width, Height)
    cv2.namedWindow('foreground', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('mask', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('applied_mask', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('cur_background', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('original_background', cv2.WINDOW_GUI_NORMAL)

    cv2.resizeWindow('foreground', window_size)
    cv2.resizeWindow('mask', window_size)
    cv2.resizeWindow('cur_background', window_size)
    cv2.resizeWindow('original_background', window_size)
    cv2.resizeWindow('applied_mask', window_size)

    window_locations = {
        'mask': (Width, -55),
        'foreground': (0,-55),
        'original_background': (0, Height),
        'cur_background': (Width, Height),
        'applied_mask': (Width*2, Height)
    }
    orig_bg = bg_subtractor.getBackgroundImage()
    empty = np.zeros_like(orig_bg)

    cv2.imshow('original_background', orig_bg)
    cv2.imshow('cur_background', empty)
    cv2.imshow('foreground', empty)
    cv2.imshow('mask', empty)
    cv2.moveWindow('original_background', *window_locations['original_background'])
    cv2.moveWindow('mask', *window_locations['mask'])
    cv2.moveWindow('foreground', *window_locations['foreground'])
    cv2.moveWindow('cur_background', *window_locations['cur_background'])
    cv2.moveWindow('applied_mask', *window_locations['applied_mask'])

    for i in tqdm(range(num_frames)):
        ret, frame = vc.read()
        if not ret:
            print('failed to read next frame')
            exit(1)
        set_roi(frame, offset_y, offset_y, offset_x, offset_x)
        fg_mask = bg_subtractor.apply(frame, learningRate=learningRate)
        applied_mask = fg_mask.copy()

        foreground = frame[applied_mask > 0]

        cv2.imshow('foreground', foreground)
        cv2.imshow('mask', fg_mask)
        cv2.imshow('cur_background', bg_subtractor.getBackgroundImage())
        cv2.imshow('applied_mask', applied_mask)

        cv2.moveWindow('original_background', *window_locations['original_background'])
        cv2.moveWindow('mask', *window_locations['mask'])
        cv2.moveWindow('foreground', *window_locations['foreground'])
        cv2.moveWindow('cur_background', *window_locations['cur_background'])
        cv2.moveWindow('applied_mask', *window_locations['applied_mask'])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    vc = cv2.VideoCapture('/home/egz01/repos/VP2023/FinalProject/Outputs/stabilized.avi')
    bg_subtractor, bg_img = train_bg_subtractor(vc, offset_y=30, offset_x=100,
                                                history=30, varThreshold=25, 
                                                detectShadows=True, learningRate=-1)
    apply_subtractor_to_video(vc, bg_subtractor, offset_y=30, offset_x=100)
    vc.release()
                         

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Stabilize a video')
    parser.add_argument('-i', '--input-path', type=str, dest='input_path', default='/home/egz01/repos/VP2023/FinalProject/Outputs/stabilized.avi')
    parser.add_argument('-bo', '--binary-output-path', type=str, dest='binary_path', default='/home/egz01/repos/VP2023/FinalProject/Outputs/binary.avi')
    parser.add_argument('-eo', '--output-path', type=str, dest='extracted_path', default='/home/egz01/repos/VP2023/FinalProject/Outputs/extracted.avi')
    parser.add_argument('-w', '--show-work', action='store_true', dest='show_work')
    args = parser.parse_args()

    #bg_subtraction_block(args.input_path, args.extracted_path, args.binary_path, show_work=args.show_work)

    vc = cv2.VideoCapture(args.input_path)
    if not vc.isOpened():
        print('failed to open video')
        exit(1)
    params = VideoParameters(vc)

    cv2.namedWindow('Frame', cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('Frame', width=600, height=350)
    cv2.moveWindow('Frame', 0, -55)
    
    cv2.namedWindow('Foreground', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Foreground', width=600, height=350)
    cv2.moveWindow('Foreground', 680, -55)

    cv2.namedWindow('Background', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Background', width=600, height=350)
    cv2.moveWindow('Background', 680, 350)

    cv2.namedWindow('Original Background', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Background', width=600, height=350)
    

    offset_y = 30
    offset_x = 100

    # start by learning the background
    background_sbutractor_params = {
        #'history': 100,
        #'varThreshold': 50,
        'history': 205, #num_iterations*params.NumFrames,
        'varThreshold': 50,
        'detectShadows': True,
        'learningRate': 1,
    }
    print("Learning background...")
    bg_subtractor, bg_img = train_bg_subtractor(vc, offset_x=offset_x, 
                                                offset_y=offset_y,
                                                blur_kernel_size=(5,5),
                                                **background_sbutractor_params)
    print("Ready to tackle the video!")

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.imshow('Original Background', bg_img)
    cv2.moveWindow('Original Background', 0, 350)

    for i in tqdm(range(params.NumFrames)):
        ret, frame = vc.read()
        if frame is None:
            break
        set_roi(frame, offset_y, offset_y, offset_x, offset_x)
        mask = bg_subtractor.apply(frame, learningRate=0.001)

        cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY, dst=mask)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        ###mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((3,3), dtype=np.uint8), iterations=5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=5)
        #
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel, iterations=5)
        ##mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=kernel, iterations=5)
        mask[mask < 250] = 0
        #mask[mask > 0] = 255

        # I expect to be left with human moving parts only here
        #retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        #
        #total_pixels = params.Height*params.Width
        #max_label = labels.max()
        #new_mask = np.zeros_like(mask)
        #'''for label, stat in zip(range(max_label), stats):
        #    left, top, width, height, area = stat
        #    if area > 300:
        #        label_shade = label/max_label * 255
        #        new_mask[labels == label] = label_shade'''
        #
        ## cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        #new_mask[labels > 0] = 255
        #
        #new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel=np.ones((7,7), dtype=np.uint8), iterations=5)
        #new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel=np.ones((3,3), dtype=np.uint8), iterations=5)
        
        # for center in centroids:
        #     x, y = center
        #     if x > 0 and y > 0:
        #         cv2.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)

        # find contours
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)'''

        '''for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)'''
        
        #binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[mask == 0] = 0
        binary_mask[mask == 255] = 1

        frame[binary_mask == 0] = [0,0,0] # blacken the background

        cv2.imshow('Frame', frame)
        cv2.moveWindow('Frame', 0, -55)
        cv2.imshow('Foreground', mask)
        cv2.moveWindow('Foreground', 680, -55)
        cv2.imshow('Background', bg_subtractor.getBackgroundImage())
        cv2.moveWindow('Background', 680, 350)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            vc.release()
            cv2.destroyAllWindows()
            break
 
vc.release()
