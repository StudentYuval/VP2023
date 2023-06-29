import cv2
import time
import numpy as np
from tqdm import tqdm
from funcs_and_structs import *
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.pyplot as plt

def calc_pdf(priors: np.ndarray, unkowns: np.ndarray):
    kde = stats.gaussian_kde(priors, bw_method = 0.5)
    pdf_values = kde.evaluate(unkowns)
    return pdf_values

def calc_alpha_for_frame(img: np.ndarray, trimap: np.ndarray) -> np.ndarray:
    img = img.astype(float) / 255
    # trimap = cv2.cvtColor(trimap, cv2.COLOR_BGR2GRAY)
    trimap = trimap.astype(float) / 255

    # Get the foreground, background and unknown regions
    fg = trimap == 1
    bg = trimap == 0
    unknown = np.logical_and(trimap > 0, trimap < 1)
    
    background_samples = img[bg]
    foreground_samples = img[fg]
    if len(background_samples) == 0 or len(foreground_samples) == 0:
        return trimap

    # nonzero samples
    background_samples = background_samples[np.nonzero(background_samples)[0]]
    foreground_samples = foreground_samples[np.nonzero(foreground_samples)[0]]
    
    # factor both to about a 1000 samples
    if len(background_samples) > 1000:
        background_samples = background_samples[::int(len(background_samples)/1000)]
    if len(foreground_samples) > 1000:
        foreground_samples = foreground_samples[::int(len(foreground_samples)/1000)]

    # Get the probability density function for foreground and background
    datapoints_coords = np.nonzero(unknown)
    unknown_datapoints = img[datapoints_coords]
    Pbg = calc_pdf(background_samples.T, unknown_datapoints.T)
    Pfg = calc_pdf(foreground_samples.T, unknown_datapoints.T)
    prob = Pfg / (Pfg + Pbg)

    alpha = np.zeros(trimap.shape)
    alpha[datapoints_coords] = prob
    alpha[fg] = 1
    alpha[bg] = 0

    return alpha

def perform_alpha_blending(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_f = fg * alpha[:, :, np.newaxis]
    alpha_b = bg * (1 - alpha[:, :, np.newaxis])

    blended = cv2.add(alpha_f, alpha_b)
    return blended

def matting_block(extracted_path, binary_path, background_path, matted_path, alpha_path, show_work=False) -> tuple[float, float]:
    print("Matting...")

    total_time_for_matting = 0
    time_for_alpha = 0

    start_time = time.time()

    vc_extracted = cv2.VideoCapture(extracted_path)
    vc_binary = cv2.VideoCapture(binary_path)
    background = cv2.imread(background_path)
    params = VideoParameters(vc_extracted)
    cv2.resize(background, (params.Width, params.Height), background)

    vw_matted = open_writer(matted_path, params)
    vw_alpha = open_writer(alpha_path, params)
    
    if show_work:
        cv2.namedWindow('matted', cv2.WINDOW_NORMAL)
        cv2.namedWindow('alpha', cv2.WINDOW_NORMAL)
        cv2.namedWindow('trimap', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('matted', 400, 250)
        cv2.resizeWindow('alpha', 400, 250)
        cv2.resizeWindow('trimap', 400, 250)

    for i in tqdm(range(params.NumFrames)):
        _, extracted_frame = vc_extracted.read()
        _, mask_frame = vc_binary.read()
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        mask_frame[mask_frame > 0] = 255
        mask_frame = mask_frame.astype(np.uint8)
        trimap = mask_frame.copy()

        # label the unknown regions by dilating and eroding the mask
        # essentially getting the borders of the masked shape
        dilated = cv2.dilate(mask_frame, np.ones((5,5), np.uint8), iterations=3)
        eroded = cv2.erode(mask_frame, np.ones((5,5), np.uint8), iterations=3)
        undecided = cv2.add(
            cv2.bitwise_xor(mask_frame, eroded),
            cv2.bitwise_xor(mask_frame, dilated)
        )
        undecided[undecided < 255] = 0
        undecided = cv2.dilate(undecided, np.ones((3,3), np.uint8), iterations=3)
        undecided = cv2.erode(undecided, np.ones((5,5), np.uint8), iterations=3)
        trimap[undecided == 255] = 127

        # trimap should be grayscale, extracted_frame should be BGR
        start_alpha = time.time()
        alpha = calc_alpha_for_frame(extracted_frame, trimap)
        end_alpha = time.time()
        time_for_alpha += end_alpha - start_alpha

        matted = perform_alpha_blending(extracted_frame, background, alpha)

        # save to video        
        matted = np.clip(matted, 0, 255)
        matted = matted.astype(np.uint8)

        alpha = (alpha * 255).astype(np.uint8) # normalize alpha to 0-255
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR) # convert to 3 channels

        vw_matted.write(matted)
        vw_alpha.write(alpha)

        
        if show_work:
            cv2.imshow('matted', matted)
            cv2.imshow('alpha', alpha)
            cv2.imshow('trimap', trimap)
            cv2.moveWindow('matted', 100, 100)
            cv2.moveWindow('alpha', 100, 400)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(0)
    
    end_time = time.time()
    total_time_for_matting = end_time - start_time
    total_time_for_matting = total_time_for_matting - time_for_alpha

    vc_binary.release()
    vc_extracted.release()
    vw_matted.release()
    vw_alpha.release()
    cv2.destroyAllWindows()
    
    return total_time_for_matting, time_for_alpha

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--extracted-path', type=str, help='path to the extracted frames', dest='extracted_path', default='Outputs/extracted.avi')
    parser.add_argument('-b', '--binary-path', type=str, help='path to the binary frames', dest='binary_path', default='Outputs/binary.avi')
    parser.add_argument('-bg', '--background-path', type=str, help='path to the background frames', dest='background_path', default='Inputs/background.jpg')
    parser.add_argument('-m', '--matted-path', type=str, help='path to the matted frames', dest='matted_path', default='Outputs/matted.avi')
    parser.add_argument('-a', '--alpha-path', type=str, help='path to the alpha frames', dest='alpha_path', default='Outputs/alpha.avi')
    parser.add_argument('-w', '--show-work', action='store_true', help='displays visualizations of the work done by the algorithm', dest='show_work')
    args = parser.parse_args()

    matting_block(**vars(args))
