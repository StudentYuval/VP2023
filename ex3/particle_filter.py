import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.random import choice
from scipy.stats import norm


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


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    # Progress the state with time
    s_prior = s_prior.astype(float)
    state_drifted = np.copy(s_prior)
    
    drift_factor = 1.2  # Adjust the drift factor as needed
    
    # Estimate velocity from previous and current positions
    prev_positions = s_prior[:2] - s_prior[4:]
    curr_positions = s_prior[:2]
    velocity = curr_positions - prev_positions
    
    state_drifted[:2] += drift_factor * velocity

    # Add noise with different variances for X and Y axes
    noise_x = np.random.normal(0, 0.2, size=s_prior[0].shape)  # Larger variance for X
    noise_y = np.random.normal(0, 0.05, size=s_prior[1].shape)  # Smaller variance for Y
    state_drifted[0] += noise_x
    state_drifted[1] += noise_y
    return state_drifted.astype(int)


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    x, y, w, h, _, _ = state.astype(int)
    # Crop the image
    cropped_image = image[y-h:y+h, x-w:x+w]
    # Compute histogram
    hist = cv2.calcHist([cropped_image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    # Normalize
    hist = hist / hist.sum()
    return hist.flatten()



def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    normalized_cdf = cdf / np.sum(cdf)  # Normalize the probabilities
    indices = np.arange(previous_state.shape[1])
    new_indices = choice(indices, size=len(indices), p=normalized_cdf)
    
    # Perform adaptive resampling
    effective_particle_count = 1. / np.sum(np.square(cdf))
    resampling_threshold = N / 2  # Adjust the resampling threshold as needed
    
    if effective_particle_count < resampling_threshold:
        # Perform resampling
        s_next = previous_state[:, new_indices]
    else:
        # No resampling required, use the original particles
        s_next = previous_state
    
    return s_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    return -np.log(np.sum(np.sqrt(p * q)))



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
    rect = patches.Rectangle((x_avg-w_avg, y_avg-h_avg), 2*w_avg, 2*h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    max_state = state[:, np.argmax(W)]
    (x_max, y_max, w_max, h_max, _, _) = max_state
    rect = patches.Rectangle((x_max-w_max, y_max-h_max), 2*w_max, 2*h_max, linewidth=1, edgecolor='r', facecolor='none')
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
    weights = np.exp(-weights)
    weights /= weights.sum()
    
    # Initialize the variable W with the computed weights
    W = weights

    # COMPUTE CDF
    cdf = np.cumsum(weights)

    # sample particles
    S = sample_particles(S, cdf)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, cdf)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
