import cv2
import time
import numpy as np
from tqdm import tqdm
from funcs_and_structs import *

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

def find_mask():
    mask = bg_sub.apply(warped_gray)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, kernel=np.ones((5,5), dtype=np.uint8), iterations=5)
    mask = cv2.dilate(mask, kernel=np.ones((31,31), dtype=np.uint8), iterations=5)
    mask = 255 - mask

    mask[:y_offset_start, :] = 0
    mask[y_offset_end:, :] = 0
    mask[:, :x_offset_start] = 0
    mask[:, x_offset_end:] = 0

    



if __name__ == '__main__':
    stabilized_path = '/home/egz01/repos/VP2023/FinalProject/Outputs/stabilized_206299463_312497084.avi'
    extracted_path = '/home/egz01/repos/VP2023/FinalProject/Outputs/extracted_206299463_312497084.avi'
    binary_path = '/home/egz01/repos/VP2023/FinalProject/Outputs/binary_206299463_312497084.avi'
    #bg_subtraction_block(stabilized_path, extracted_path, binary_path, show_work=True)

    backSub = cv2.createBackgroundSubtractorMOG2()
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(stabilized_path))
    if not capture.isOpened():
        print('Unable to open: ' + input)
        exit(0)

    cv2.namedWindow('Frame', cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('Frame', width=700, height=450)
    
    cv2.namedWindow('FG Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FG Mask', width=700, height=450)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        cv2.imshow('Frame', frame)
        cv2.moveWindow('Frame', -40, -55)
        
        cv2.imshow('FG Mask', fgMask)
        cv2.moveWindow('FG Mask', 680, -55)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
 
    

 
 
 
 
 
 