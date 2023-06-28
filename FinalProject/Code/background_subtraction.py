import cv2
import time
import numpy as np
from tqdm import tqdm
from funcs_and_structs import *

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
                             learningRate: float=0.9,
                             reverse: bool=False,
                             history: int=30,
                             varThreshold: int=50, 
                             detectShadows: bool=False) -> tuple[cv2.BackgroundSubtractorMOG2, np.ndarray]:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
    #bg_subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=varThreshold, detectShadows=detectShadows)
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in tqdm(range(num_frames)):
        if reverse:
            vc.set(cv2.CAP_PROP_POS_FRAMES, num_frames-i-1)
        ret, frame = vc.read()
        if not ret:
            print('failed to read next frame')
            exit(1)
        set_roi(frame, offset_y, offset_y, offset_x, offset_x)
        _ = bg_subtractor.apply(frame, learningRate=learningRate)

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return bg_subtractor, bg_subtractor.getBackgroundImage()

def init_windows(shape):
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

    empty = np.zeros(shape, dtype=np.uint8)
    cv2.imshow('original_background', empty)
    cv2.imshow('cur_background', empty)
    cv2.imshow('foreground', empty)
    cv2.imshow('mask', empty)
    cv2.imshow('applied_mask', empty)

    
    window_locations = {
        'mask': (Width, -55),
        'foreground': (0,-55),
        'original_background': (0, Height),
        'cur_background': (Width, Height),
        'applied_mask': (Width*2, Height)
    }

    return  window_locations

def place_windows(window_locations):
    cv2.moveWindow('original_background', *window_locations['original_background'])
    cv2.moveWindow('mask', *window_locations['mask'])
    cv2.moveWindow('foreground', *window_locations['foreground'])
    cv2.moveWindow('cur_background', *window_locations['cur_background'])
    cv2.moveWindow('applied_mask', *window_locations['applied_mask'])


def apply_subtractor_to_video(vc: cv2.VideoCapture, 
                              rev_bg_subtractor: cv2.BackgroundSubtractorMOG2,
                              fwd_bg_subtractor: cv2.BackgroundSubtractorMOG2,
                              vw_binary: cv2.VideoWriter,
                              vw_extracted: cv2.VideoWriter,
                              learningRate=0.001,
                              offset_y: int=30, offset_x: int=100,
                              show_work: bool=False,):
    
    params = VideoParameters(vc)

    vc.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if show_work:
        window_locations = init_windows([params.Width, params.Height])
        place_windows(window_locations)
    
    close_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    erode_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dial_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    open_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    min_blob_area = 0.02*0.05*params.Height*params.Width
    #min_blob_area = 0.02*0.05*1080*1920

    for i in tqdm(range(params.NumFrames)):
        ret, frame = vc.read()
        if not ret:
            print('failed to read next frame')
            exit(1)
        set_roi(frame, offset_y, offset_y, offset_x, offset_x)
        if i < params.NumFrames/2:
            fg_mask = fwd_bg_subtractor.apply(frame, learningRate=learningRate)
        else:
            fg_mask = rev_bg_subtractor.apply(frame, learningRate=learningRate)

        applied_mask = fg_mask.copy()

        # try to merge false shades
        cv2.GaussianBlur(applied_mask, (15,15), 0, dst=applied_mask)
        cv2.threshold(applied_mask, 127, 255, cv2.THRESH_BINARY, dst=applied_mask)
        applied_mask[applied_mask < 255] = 0

        applied_mask = cv2.morphologyEx(applied_mask, cv2.MORPH_OPEN, open_elem)
        applied_mask = cv2.morphologyEx(applied_mask, cv2.MORPH_CLOSE, close_elem)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(applied_mask, connectivity=8)
        max_label = labels.max()
        for label, stat in zip(range(0,max_label+1), stats):
            x, y, height, width, area = stat
            if area < min_blob_area:
                applied_mask[labels == label] = 0

        for i in range(3):
            applied_mask = cv2.morphologyEx(applied_mask, cv2.MORPH_DILATE, dial_elem, iterations=1)
            applied_mask = cv2.morphologyEx(applied_mask, cv2.MORPH_CLOSE, close_elem, iterations=4)
        
        applied_mask = cv2.morphologyEx(applied_mask, cv2.MORPH_ERODE, erode_elem, iterations=5)

        foreground = np.zeros_like(frame)
        foreground[applied_mask == 255] = frame[applied_mask == 255]

        vw_binary.write(cv2.cvtColor(applied_mask, cv2.COLOR_GRAY2BGR))
        vw_extracted.write(foreground)

        if show_work:
            cv2.imshow('foreground', foreground)
            cv2.imshow('mask', fg_mask)
            if i < params.NumFrames/2:
                cv2.imshow('cur_background', fwd_bg_subtractor.getBackgroundImage())
            else:
                cv2.imshow('cur_background', rev_bg_subtractor.getBackgroundImage())
            cv2.imshow('applied_mask', applied_mask)
            
            place_windows(window_locations)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                # pause
                cv2.waitKey(0)

    cv2.destroyAllWindows()

def bg_subtraction_block(stabilized_path: str, 
                         extracted_path: str, binary_path: str, show_work=False) -> float:
    print("Extracting foreground...")

    start_time = time.time()

    vc = cv2.VideoCapture(stabilized_path)
    params = VideoParameters(vc)

    vw_extracted = open_writer(extracted_path, params)
    vw_binary = open_writer(binary_path, params)

    offset_y = round(0.027*params.Height)
    offset_x = round(0.052*params.Width)

    history_length = params.NumFrames//3

    rev_bg_subtractor, _ = train_bg_subtractor(vc, offset_y=30, offset_x=100,
                                                history=history_length, varThreshold=40, 
                                                detectShadows=True, learningRate=-1, reverse=True)
    fwd_bg_subtractor, _ = train_bg_subtractor(vc, offset_y=30, offset_x=100,
                                                history=history_length, varThreshold=40,
                                                detectShadows=True, learningRate=-1, reverse=False)

    apply_subtractor_to_video(vc ,rev_bg_subtractor, fwd_bg_subtractor, 
                              vw_binary, vw_extracted, offset_y=offset_y, 
                              offset_x=offset_x, learningRate=0,
                              show_work=show_work)

    end_time = time.time()

    vc.release()
    vw_extracted.release()
    vw_binary.release()
    cv2.destroyAllWindows()

    return end_time - start_time

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stabilized-path', type=str, default='Outputs/stabilized.avi')
    parser.add_argument('-e', '--extracted-path', type=str, default='Outputs/extracted.avi')
    parser.add_argument('-b', '--binary-path', type=str, default='Outputs/binary.avi')
    parser.add_argument('-w', '--show-work', action='store_true', dest='show_work', default=False)
    args = parser.parse_args()

    bg_subtraction_block(args.stabilized_path, args.extracted_path, args.binary_path, show_work=args.show_work)
 