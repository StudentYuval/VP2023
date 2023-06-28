import cv2
import time
import numpy as np
from tqdm import tqdm
from funcs_and_structs import *
import matplotlib.pyplot as plt

def stabilization_block(input_path: str, output_path: str, show_work: bool = False) -> float:
    print("Stabilizing video...")
    
    start_time = time.time()
    vc = cv2.VideoCapture(input_path)
    ret, prev_frame = vc.read()
    first_frame = prev_frame.copy()

    if not ret:
        print('failed to read first frame')
        exit(1)

    params = VideoParameters(vc)
    vw = open_writer(output_path, params)
    vw.write(prev_frame)

    optical_flow_params = {
                    "winSize":(7,7),
                    "maxLevel":5,
                    "criteria":(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 5, 0.0001)
            }

    y_offset_start = round(params.Height*0.03)
    y_offset_end = round(params.Height*0.99)
    x_offset_start = round(params.Width*0.03)
    x_offset_end = round(params.Width*0.99)
    warped_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    warped = prev_frame.copy()
    borders_estimator = np.zeros_like(warped, dtype=np.uint8)
    disp_features = None

    if show_work:
        prev_orig_frame = prev_frame.copy()

        cv2.namedWindow('showing_work', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('showing_work', width=1000, height=300)

        avg_stab_mse = 0
        avg_orig_mse = 0

    for i in tqdm(range(params.NumFrames-1)):
        ret, cur_frame = vc.read()
        if not ret:
            print('failed to read next frame')
            break
        
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        orig_features = find_points_to_track(warped_gray)
        disp_features, st, err = cv2.calcOpticalFlowPyrLK(warped_gray, cur_gray, orig_features, disp_features, **optical_flow_params)

        distances = np.sqrt(np.power(orig_features - disp_features,2).sum(axis=2))
        mean_distance = distances[st == 1].mean()
        threshold = 1.5*mean_distance # best is 1.5 the average distance between points!
        disp_points = disp_features[(st == 1) & (distances <= threshold)]
        orig_points = orig_features[(st == 1) & (distances <= threshold)]
        
        M, used_points = cv2.findHomography(disp_points, orig_points, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        warped = cv2.warpPerspective(cur_frame, M, (params.Width, params.Height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        warped = warped[y_offset_start:y_offset_end, x_offset_start:x_offset_end]
        warped = cv2.resize(warped, dsize=(params.Width, params.Height))
        
        if show_work:
            stab_mse = calc_mse_between_frames(prev_frame, warped)
            avg_stab_mse = (avg_stab_mse*i + stab_mse)/(i+1)

            orig_mse = calc_mse_between_frames(prev_orig_frame, cur_frame)
            avg_orig_mse = (avg_orig_mse*i + orig_mse)/(i+1)

            prev_orig_frame = cur_frame.copy()

        vw.write(warped)
        prev_frame = warped.copy()
        
        # for debugging, might use it for the report
        if show_work:
            frame = cur_frame.copy()
            for disp, orig in zip(disp_points, orig_points):
                disp_point = disp.ravel().astype(np.int32)
                orig_point = orig.ravel().astype(np.int32)
                frame = cv2.line(frame, disp_point, orig_point, (0, 255, 0), 2)
                frame = cv2.circle(frame, disp_point, 3, (0, 0, 255), -1)
                frame = cv2.circle(frame, orig_point, 3, (255, 0, 0), -1)

            frame = cv2.putText(frame, f"Average MSE: {avg_orig_mse:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, f"Current MSE: {orig_mse:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frame = cv2.putText(frame, f"Num used features: {np.count_nonzero(used_points):.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            warped_show = cv2.putText(warped, f"Average MSE: {avg_stab_mse:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            warped_show = cv2.putText(warped_show, f"Current MSE: {stab_mse:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            to_show = np.concatenate((cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.resize(warped_show, (0, 0), fx=0.5, fy=0.5)), axis=1)
            cv2.imshow('showing_work', to_show)
            cv2.moveWindow('showing_work', 0, 0)

            key_input = cv2.waitKey(1) & 0xFF
            if key_input == ord('q'):
                print("window closed by user")
                cv2.destroyWindow('showing_work')
                show_work = False
                break
            elif key_input == ord('c'):
                print("Terminated by user")
                cv2.destroyWindow('showing_work')
                show_work = False
                

    vc.release()
    vw.release()
    cv2.destroyAllWindows()
    
    print(f"Finished stabilizing video")
    
    if show_work:
        print(f"Average MSE of original video: {avg_orig_mse:.2f}")
        print(f"Average MSE of stabilized video: {avg_stab_mse:.2f}")

    return time.time() - start_time

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Stabilize a video')
    parser.add_argument('-i', '--input-path', type=str, dest='input_path', default='/home/egz01/repos/VP2023/FinalProject/Inputs/INPUT.avi')
    parser.add_argument('-o', '--output-path', type=str, dest='output_path', default='/home/egz01/repos/VP2023/FinalProject/Outputs/stabilized.avi')
    parser.add_argument('-w', '--show-work', action='store_true', dest='show_work')
    args = parser.parse_args()

    stabilization_block(args.input_path, args.output_path, args.show_work)
    
