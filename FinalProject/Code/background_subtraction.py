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

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

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

if __name__ == '__main__':
    stabilized_path = '/home/egz01/repos/VP2023/FinalProject/Outputs/stabilized_206299463_312497084.avi'
    extracted_path = '/home/egz01/repos/VP2023/FinalProject/Outputs/extracted_206299463_312497084.avi'
    binary_path = '/home/egz01/repos/VP2023/FinalProject/Outputs/binary_206299463_312497084.avi'
    bg_subtraction_block(stabilized_path, extracted_path, binary_path, show_work=True)
    print("Done")
 
 