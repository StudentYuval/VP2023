import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.animation as animation

# change IDs to your IDs.
ID1 = '206299463'
ID2 = '312497084'

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y

# state index constants for readability becuase I keep forgetting
X_ind = 0
Y_ind = 1
W_ind = 2
H_ind = 3
VX_ind = 4
VY_ind = 5

WIDTH = 576
HEIGHT = 352

# set this to True if you want to generate a video of the tracking process
GENERATE_VIDEO = False

def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    
    ''' A little about our motion model assumptions:
     the input video is of a running human, mostly on a horizontal plane; for an average running pace of 2~3 m/s, 
     a running human moves (horizontally) an average of 0.1 meters between frames. 
     We estimated the FPS of the input video to be ~25[fps] by by observing the video at different frame rates 
     and converging on what felt like natural movement; we then estimate the pixel/meter ratio to be ~65
     by measuring the human's height in pixels (110[px]) and dividing it by an average human's height (1.7[m]),
     as well as assuming same px/m ratio for horizontal and vertical directions.
     Finally, we end up with a possible range 6~7 pixels horizontal displacement between frames; 
     as for vertical displacement, assuming a constant height human (most likely), we assume a maximum of
     ~3px vertical displacement between frames, for the scenario that the human meets a sudden slope.'''

    # Progress the state with time
    s_prior = s_prior.astype(float)
    state_drifted = np.copy(s_prior)
    
    # update current state's positions according to the prior's velocity
    state_drifted[[X_ind, Y_ind]] += s_prior[[VX_ind, VY_ind]]
    
    # the bounding box might drift out of the frame if the
    # tracked object is moving towards the frame's edges
    state_drifted[[X_ind, Y_ind]] = np.clip(state_drifted[[X_ind, Y_ind]].T, [0, 0], [WIDTH-1, HEIGHT-1]).T

    # estimating uniform noise: x,y limits according typical human velocities,
    # vx, vy limits according to typical human acceleration - a human might start/stop moving
    # which will be reflected in abrupt changes in velocity
    x_lim = 7
    y_lim = 3
    vx_lim = 4
    vy_lim = 2
    h_lim = w_lim = 0 # no changes in width/height
    lims = np.vstack(np.array([x_lim, y_lim, w_lim, h_lim, vx_lim, vy_lim]))

    noise = np.random.uniform(-1*lims, lims, size=state_drifted.shape)
    state_drifted += noise

    # keep velocities within reasonable limits as described in the motion model
    state_drifted[[VX_ind, VY_ind]] = np.clip(state_drifted[[VX_ind, VY_ind]].T, [-1.2*vx_lim, -0.8*vy_lim], [1.2*vx_lim, 0.8*vy_lim]).T
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    x, y, w, h, _, _ = state.astype(int)
    patch = image[y-h:y+h, x-w:x+w]
    H = cv2.calcHist([patch], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    H /= H.sum() # normalize histogram
    return H.flatten()


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    rs = np.random.random(size=cdf.shape)
    diffs = cdf - np.vstack(rs) # The resultant matrix D(iffs) holds: Dij = cdf[j]-rs[i]
    diffs[diffs <= 0] = np.inf   # I eliminate all the negative values in diffs from the comparison
    new_indices = diffs.argmin(axis=1) # find the minimum value in each column, that is the new index of the particle
    s_next = previous_state[:, new_indices]

    # purposefully not calculating velocity after resampling
    # instead, rely on the motion model described in predict_particles
    # to estimate the velocity

    return s_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    return np.exp(20*np.sqrt(p*q).sum())


def create_image_with_boundingbox(image: np.ndarray, 
                                  mean_bbox: tuple, 
                                  max_bbox: tuple,
                                  current_bbox: tuple
                                  ) -> np.ndarray:
    """Create an image with the bounding box and the ID.
       I used this function to create a video of the entire tracking process,
       this proved very helpful in estimating the best tuning parameters for 
       the tracking algorithm, as well as include tweaks in the algorithm itself.
       To view the video, set the "GENERATE_VIDEO" variable to True at the top of the file
        
    """
    image_with_bbox = image.copy()

    # max bbox in red
    x, y, w, h = [int(round(i)) for i in max_bbox]
    image_with_bbox = cv2.rectangle(image_with_bbox, (x-w, y-h), (x+w, y+h), (0, 0, 255), 2)

    # mean bbox in green
    x, y, w, h = [int(round(i)) for i in mean_bbox]
    image_with_bbox = cv2.rectangle(image_with_bbox, (x-w, y-h), (x+w, y+h), (0, 255, 0), 2)

    # current bbox in blue
    x, y, w, h = [int(round(i)) for i in current_bbox]
    image_with_bbox = cv2.rectangle(image_with_bbox, (x-w, y-h), (x+w, y+h), (255, 0, 0), 2)

    return image_with_bbox


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame number = " + str(frame_index))

    # Avg particle box
    avg_state = np.average(state, axis=1, weights=W)
    (x_avg, y_avg, w_avg, h_avg, _, _) = avg_state
    rect = patches.Rectangle((x_avg-w_avg, y_avg-h_avg), 2*w_avg, 2*h_avg, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    max_state = state[:, np.argmax(W)]
    (x_max, y_max, w_max, h_max, _, _) = max_state
    rect = patches.Rectangle((x_max-w_max, y_max-h_max), 2*w_max, 2*h_max, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, np.array(s_initial))
    
    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    weights = np.array([bhattacharyya_distance(compute_normalized_histogram(image, s), q) for s in S.T])
    weights /= weights.sum()
    
    # Initialize the variable W with the computed weights
    W = weights

    # COMPUTE CDF
    cdf = np.cumsum(weights)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    images_paths = [os.path.join(IMAGE_DIR_PATH, image_name) for image_name in image_name_list]
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}

    if GENERATE_VIDEO:
        dimensions = image.shape[:2][::-1]
        slowed_down_vw = cv2.VideoWriter(os.path.join(RESULTS, "slowed_down_video.avi"),
                             fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                             fps=10,
                             frameSize=dimensions,
                             isColor=True)

        real_time_vw = cv2.VideoWriter(os.path.join(RESULTS, "normal_speed_video.avi"),
                             fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                             fps=25,
                             frameSize=dimensions,
                             isColor=True)

        mean_bbox = s_initial[:4]
        max_bbox = s_initial[:4]

    for image_path in images_paths[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, cdf)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE)
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        weights = np.array([bhattacharyya_distance(compute_normalized_histogram(current_image, s), q) for s in S.T])
        weights /= weights.sum()
        W = weights

        # COMPUTE CDF
        cdf = np.cumsum(weights)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)
            if GENERATE_VIDEO:
                mean_bbox = frame_index_to_avg_state[images_processed]
                max_bbox = frame_index_to_max_state[images_processed]

        if GENERATE_VIDEO:
            current_frame_bbox = np.average(S[[X_ind, Y_ind, W_ind, H_ind]], axis=1, weights=W)
            bounded_frame = create_image_with_boundingbox(current_image, mean_bbox, max_bbox, current_frame_bbox)
            slowed_down_vw.write(bounded_frame)
            real_time_vw.write(bounded_frame)
    
    if GENERATE_VIDEO:
        slowed_down_vw.release()
        real_time_vw.release()
            
    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
