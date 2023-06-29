import cv2
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from funcs_and_structs import *
from background_subtraction import train_bg_subtractor
import time

def tracking_block(matted_path: str, output_path: str, tracking_json: OrderedDict, show_work: bool) -> float:
    print("Tracking...")
    start_time = time.time()

    vc = cv2.VideoCapture(matted_path)
    params = VideoParameters(vc)

    vw = open_writer(output_path, params)
    
    bg_subtractor, _ = train_bg_subtractor(vc, offset_y=0, offset_x=0,
                        history=params.NumFrames, varThreshold=25,
                        detectShadows=False, learningRate=-1, reverse=False)

    if show_work:
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking", 400, 350)
    prev_contour = None
    for i in tqdm(range(params.NumFrames)):
        ret, frame = vc.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame, learningRate=-1)

        # remove noise
        kernel = np.ones((3,3),np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours in the mask
        contours, hierarchy = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding boxes for each contour
        max_area = params.Height*params.Width
        largest = 0
        for contour in contours:
            # get size of bounding box
            bbox_size = cv2.contourArea(contour)
            if bbox_size > largest and bbox_size < max_area:
                largest = bbox_size
                largest_contour = contour

        try:
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
            tracking_json[i] = [x, y, w, h]
            prev_contour = largest_contour
        except:
            if prev_contour is not None:
                (x, y, w, h) = cv2.boundingRect(prev_contour)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
                tracking_json[i] = [x, y, w, h]
            else:
                tracking_json[i] = [0, 0, 0, 0]
        
        if show_work:
            cv2.imshow("Tracking", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
        
        vw.write(frame)

    end_time = time.time()
    vw.release()
    vc.release()
    cv2.destroyAllWindows()
    return end_time - start_time

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--matted-path', type=str, default='Outputs/matted.avi')
    parser.add_argument('-o', '--output-path', type=str, default='Outputs/OUTPUT.avi')
    parser.add_argument('-w', '--show-work', action='store_true', dest='show_work', default=False)
    args = parser.parse_args()

    tracking_block(args.matted_path, args.output_path, None, args.show_work)

