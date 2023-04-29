import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
import time

# FILL IN YOUR ID
ID1 = '206299463'
ID2 = '312497084'


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5

def fourcc_to_str(fourcc: int):
    fourcc_str = ''
    while fourcc > 0:
        fourcc_str += chr(fourcc & 0xFF)
        fourcc >>= 8
    return fourcc_str


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int, boundary='symm', mode='same') -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    cur_image: np.ndarray = image.copy().astype(np.float64)
    pyramid = [cur_image]
    
    for i in range(num_levels):
        cur_image = signal.convolve2d(cur_image, PYRAMID_FILTER, boundary=boundary, mode=mode)
        cur_image = cur_image[::2, ::2]
        pyramid.append(cur_image.copy())

    return pyramid

def show_pyramid(pyramid: list[np.ndarray], num_levels: int, image_id: int):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, num_levels + 1, tight_layout=True, figsize=(10, 5))
    axs = axs.flatten()
    for i in range(num_levels + 1):
        axs[i].imshow(pyramid[i], cmap='gray')
        axs[i].axis(False)
        axs[i].set_title(f'Level {i}')
    fig.suptitle(f"Pyramid progression of $I_{image_id}$")    
    # fig.savefig(f'pyramid_{image_id}.png', dpi=300)
    plt.show()

def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel's neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    du = np.zeros(I1.shape, dtype=np.float64)
    dv = np.zeros(I1.shape, dtype=np.float64)

    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, boundary='symm', mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, boundary='symm', mode='same')
    It = I2 - I1

    nrows = I1.shape[0]
    ncols = I1.shape[1]
    for row in range(window_size//2, nrows-window_size//2-1):
        for col in range(window_size//2, ncols-window_size//2-1):
            Ix_window = Ix[row-window_size//2:row+window_size//2+1, col-window_size//2:col+window_size//2+1]
            Iy_window = Iy[row-window_size//2:row+window_size//2+1, col-window_size//2:col+window_size//2+1]
            It_window = It[row-window_size//2:row+window_size//2+1, col-window_size//2:col+window_size//2+1]

            A = np.vstack((Ix_window.flatten(), Iy_window.flatten())).T
            b = -1 * It_window.flatten()
            
            try:
                A_T_A = A.T @ A
                A_T_b = A.T @ b
              
                tau = 1e8
                cond_num = np.linalg.cond(A_T_A)
                
                if cond_num > tau:
                    u, v = (0, 0)
                else:
                    A_T_A_inv = np.linalg.inv(A_T_A)
                    u, v = A_T_A_inv @ A_T_b
                
            except:
                u, v = (0, 0)

            du[row, col] = u
            dv[row, col] = v

    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray, mode: str = 'remap') -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    image_warp = np.zeros(image.shape, dtype=np.float64)
    if mode == 'remap':
        h = image.shape[0]
        w = image.shape[1]

        # resize u,v to the shape of the image
        if u.shape != image.shape:
            factor_u = w / u.shape[1]
            factor_v = h / v.shape[0]
            u = factor_u*cv2.resize(u, (w, h))
            v = factor_v*cv2.resize(v, (w, h))

        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + u).astype(np.float32)
        map_y = (y + v).astype(np.float32)
        image_warp = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    elif mode == 'shift':
        ndimage.shift(image, (v, u), output=image_warp, order=3, mode='constant', cval=np.nan)
    else:
        raise ValueError('Invalid mode')

    # fill nan holes
    nan_inds = np.isnan(image_warp)
    image_warp[nan_inds] = image[nan_inds]

    return image_warp

def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    I1_ = I1.copy().astype(np.float64)
    I2_ = I2.copy().astype(np.float64)
    
    # resize I1, I2 to prepare for the decimations pyramid:
    K = int(np.ceil(I1_.shape[0]/(2**(num_levels - 1 + 1))))
    M = int(np.ceil(I1_.shape[1]/(2**(num_levels - 1 + 1))))
    IMAGE_SIZE = (M*(2**(num_levels - 1 + 1)), K*(2**(num_levels - 1 + 1)))
    if I1_.shape != IMAGE_SIZE:
        I1_ = cv2.resize(I1, IMAGE_SIZE)
    if I2_.shape != IMAGE_SIZE:
        I2_ = cv2.resize(I2, IMAGE_SIZE)

    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1_, num_levels)
    pyramid_I2 = build_pyramid(I2_, num_levels)

    u = np.zeros(pyramid_I1[-1].shape, dtype=np.float64)
    v = np.zeros(pyramid_I1[-1].shape, dtype=np.float64)

    for i in range(num_levels, -1, -1):
        warped_i2 = warp_image(pyramid_I2[i], u, v)
        for _ in range(max_iter):
            du, dv = lucas_kanade_step(pyramid_I1[i], warped_i2, window_size)
            u += du
            v += dv
            warped_i2 = warp_image(pyramid_I2[i], u, v)
        if i != 0:
            h, w = pyramid_I2[i-1].shape
            factor_u = w / u.shape[1]
            factor_v = h / v.shape[0]
            u = factor_u*cv2.resize(u, pyramid_I2[i-1].shape[::-1], )
            v = factor_v*cv2.resize(v, pyramid_I2[i-1].shape[::-1])
    return u, v

def resize_to_fit_decimation(height: int, width: int, num_levels: int):
    ''' calculates new shape for an image that will be decimated num_levels times
        Note: the shape is in numpy format, exactly opposite of cv2 format
        I use cv2.resize(image, new_shape[::-1]) to resize the image to the new shape,
        as instructed in the recipe from the previous section
    '''
    K = int(np.ceil(height/(2**(num_levels - 1 + 1))))
    M = int(np.ceil(width/(2**(num_levels - 1 + 1))))
    return (K*(2**(num_levels - 1 + 1)), M*(2**(num_levels - 1 + 1)))
        
def read_and_fit_next_frame_from_video(vc: cv2.VideoCapture, height: int, width: int) -> np.ndarray:
    ret, frame = vc.read()
    if not ret:
        None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    frame = cv2.resize(frame, (width, height))
    return frame

def write_frame_to_video(vw: cv2.VideoWriter, frame: np.ndarray, height: int, width: int):
    frame = cv2.resize(frame, (width, height))
    vw.write(frame.astype(np.uint8))

def plot_U(U: tuple[np.ndarray, np.ndarray]):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs = axs.flatten()

    axs[0].imshow(U[0], cmap='gray')
    axs[0].set_title('u')

    axs[1].imshow(U[1], cmap='gray')
    axs[1].set_title('v')

    plt.show()

def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    vc = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vc)
    orig_height = params['height']
    orig_width = params['width']

    vw = cv2.VideoWriter(output_video_path, 
                                fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                                fps=params['fps'],
                                frameSize=(orig_width, orig_height),
                                isColor=False
                            )
    
    new_height, new_width = resize_to_fit_decimation(height=orig_height,
                                         width=orig_width,
                                         num_levels=num_levels)
    
    first_frame = read_and_fit_next_frame_from_video(vc, 
                                                     height=new_height, 
                                                     width=new_width)
    
    if first_frame is None:
        raise ValueError('Could not read first frame from video, terminating.')
    write_frame_to_video(vw, first_frame, height=orig_height, width=orig_width)

    # define region of interest
    start = window_size//2
    end = -1 * (start)

    u = np.zeros((new_height, new_width), dtype=np.float64)
    v = np.zeros((new_height, new_width), dtype=np.float64)
    prev_U = (u, v)

    prev_frame = first_frame
    for i in tqdm(range(params['frame_count'] - 1)):
    #for i in tqdm(range(10)):
        frame = read_and_fit_next_frame_from_video(vc, height=new_height, width=new_width)
        if frame is None:
            break # end of video arrived prematurely...
        
        cur_U = lucas_kanade_optical_flow(prev_frame, frame, window_size, max_iter, num_levels)
        
        cur_U[0][start:end,start:end] = cur_U[0][start:end, start:end].mean()
        cur_U[1][start:end,start:end] = cur_U[1][start:end, start:end].mean()
        prev_U[0][start:end,start:end] += cur_U[0][start:end,start:end]
        prev_U[1][start:end,start:end] += cur_U[1][start:end,start:end]

        warped_frame = warp_image(frame, prev_U[0], prev_U[1])

        write_frame_to_video(vw, warped_frame, height=orig_height, width=orig_width)
        prev_frame = frame # maybe prev_fame = warped_frame?

    vw.release()
    vc.release()
    cv2.destroyAllWindows()


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    du = np.zeros(I1.shape, dtype=np.float64)
    dv = np.zeros(I1.shape, dtype=np.float64)
    
    if I1.shape[0] * I1.shape[1] < 10000:
        return lucas_kanade_step(I1, I2, window_size)
    
    
    '''corners = cv2.cornerHarris(I2.astype(np.float32), 2, 3, 0.04)
    corners = cv2.dilate(corners, None)
    corners[corners < 0.01*corners.max()] = 0
    corners[corners.nonzero()] = 1
    corners = corners.astype(bool)
    corner_coords = np.argwhere(corners)'''

    # use of this function instead of cv2.cornerHarris yidls way better results, and is ~3x faster!
    corner_coords = cv2.goodFeaturesToTrack(I2.astype(np.float32), maxCorners=500, qualityLevel=0.01, minDistance=10)
    corner_coords = corner_coords.astype(np.int64).reshape((-1, 2))
    
    # I keep forgetting that cv2 returns (col, row) and not (row, col)
    corner_coords = corner_coords[:, ::-1]
    corners = np.zeros_like(I2).astype(bool)
    corners[corner_coords[:,0], corner_coords[:,1]] = True

    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, boundary='symm', mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, boundary='symm', mode='same')
    It = I2 - I1

    for row, col in corner_coords:
        # skip corners too close to edge
        if row - window_size//2 < 0 or row + window_size//2 + 1 > I1.shape[0] or \
           col - window_size//2 < 0 or col + window_size//2 + 1 > I1.shape[1]:
            continue

        Ix_window = Ix[row-window_size//2:row+window_size//2+1, col-window_size//2:col+window_size//2+1]
        Iy_window = Iy[row-window_size//2:row+window_size//2+1, col-window_size//2:col+window_size//2+1]
        It_window = It[row-window_size//2:row+window_size//2+1, col-window_size//2:col+window_size//2+1]

        A = np.vstack((Ix_window.flatten(), Iy_window.flatten())).T
        b = -1 * It_window.flatten()
        
        try:
            A_T_A = A.T @ A
            A_T_b = A.T @ b
          
            tau = 1e8
            cond_num = np.linalg.cond(A_T_A)
            
            if cond_num > tau:
                u, v = (0, 0)
            else:
                A_T_A_inv = np.linalg.inv(A_T_A)
                u, v = A_T_A_inv @ A_T_b
            
        except:
            u, v = (0, 0)

        du[row, col] = u
        dv[row, col] = v

    # set du, dv at valid pixels to mean of du, dv at corners
    du[window_size//2:-1*(window_size//2),window_size//2:-1*(window_size//2)] = du[corners].mean()
    dv[window_size//2:-1*(window_size//2),window_size//2:-1*(window_size//2)] = dv[corners].mean()

    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** num_levels)))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** num_levels)))
    IMAGE_SIZE = (w_factor * (2 ** num_levels),
                  h_factor * (2 ** num_levels))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyramid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1
    u = np.zeros(pyramid_I2[-1].shape)  # create u in the size of smallest image
    v = np.zeros(pyramid_I2[-1].shape)  # create v in the size of smallest image

    for i in range(num_levels, -1, -1):
        warped_i2 = warp_image(pyramid_I2[i], u, v)
        for _ in range(max_iter):
            du, dv = faster_lucas_kanade_step(pyramid_I1[i], warped_i2, window_size)
            u += du
            v += dv
            warped_i2 = warp_image(pyramid_I2[i], u, v)
        if i != 0:
            h, w = pyramid_I2[i-1].shape
            factor_u = w / u.shape[1]
            factor_v = h / v.shape[0]
            u = factor_u*cv2.resize(u, pyramid_I2[i-1].shape[::-1], )
            v = factor_v*cv2.resize(v, pyramid_I2[i-1].shape[::-1])
    return u, v

def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """

    vc = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vc)
    orig_height = params['height']
    orig_width = params['width']

    vw = cv2.VideoWriter(output_video_path, 
                                fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                                fps=params['fps'],
                                frameSize=(orig_width, orig_height),
                                isColor=False
                            )
    
    new_height, new_width = resize_to_fit_decimation(height=orig_height,
                                         width=orig_width,
                                         num_levels=num_levels)
    
    first_frame = read_and_fit_next_frame_from_video(vc, 
                                                     height=new_height, 
                                                     width=new_width)
    
    if first_frame is None:
        raise ValueError('Could not read first frame from video, terminating.')
    write_frame_to_video(vw, first_frame, height=orig_height, width=orig_width)

    # define region of interest
    start = window_size//2
    end = -1 * (start)

    u = np.zeros((new_height, new_width), dtype=np.float64)
    v = np.zeros((new_height, new_width), dtype=np.float64)
    prev_U = (u, v)

    prev_frame = first_frame
    for i in tqdm(range(params['frame_count'] - 1)):
    #for i in tqdm(range(10)):
        frame = read_and_fit_next_frame_from_video(vc, height=new_height, width=new_width)
        if frame is None:
            break # end of video arrived prematurely...
        cur_U = faster_lucas_kanade_optical_flow(prev_frame, frame, window_size, max_iter, num_levels)
        
        cur_U[0][start:end,start:end] = cur_U[0][start:end, start:end].mean()
        cur_U[1][start:end,start:end] = cur_U[1][start:end, start:end].mean()
        prev_U[0][start:end,start:end] += cur_U[0][start:end,start:end]
        prev_U[1][start:end,start:end] += cur_U[1][start:end,start:end]

        warped_frame = warp_image(frame, prev_U[0], prev_U[1])

        write_frame_to_video(vw, warped_frame, height=orig_height, width=orig_width)
        prev_frame = frame

    vw.release()
    vc.release()
    cv2.destroyAllWindows()
        


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """

    vc = cv2.VideoCapture(input_video_path)
    params = get_video_parameters(vc)
    orig_height = params['height']
    orig_width = params['width']

    vw = cv2.VideoWriter(output_video_path, 
                                fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                                fps=params['fps'],
                                frameSize=(orig_width, orig_height),
                                isColor=False
                            )
    
    new_height, new_width = resize_to_fit_decimation(height=orig_height,
                                         width=orig_width,
                                         num_levels=num_levels)
    
    first_frame = read_and_fit_next_frame_from_video(vc, 
                                                     height=new_height, 
                                                     width=new_width)
    
    if first_frame is None:
        raise ValueError('Could not read first frame from video, terminating.')
    write_frame_to_video(vw, first_frame, height=orig_height, width=orig_width)

    # define region of interest
    start = window_size//2
    end = -1 * (start)

    u = np.zeros((new_height, new_width), dtype=np.float64)
    v = np.zeros((new_height, new_width), dtype=np.float64)
    prev_U = (u, v)

    prev_frame = first_frame
    for i in tqdm(range(params['frame_count'] - 1)):
    #for i in tqdm(range(10)):
        frame = read_and_fit_next_frame_from_video(vc, height=new_height, width=new_width)
        if frame is None:
            break # end of video arrived prematurely...
        
        cur_U = faster_lucas_kanade_optical_flow(prev_frame, frame, window_size, max_iter, num_levels)
        
        cur_U[0][start:end,start:end] = cur_U[0][start:end, start:end].mean()
        cur_U[1][start:end,start:end] = cur_U[1][start:end, start:end].mean()
        prev_U[0][start:end,start:end] += cur_U[0][start:end,start:end]
        prev_U[1][start:end,start:end] += cur_U[1][start:end,start:end]

        warped_frame = warp_image(frame, prev_U[0], prev_U[1])

        warped_frame = cv2.resize(warped_frame, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
        warped_frame = warped_frame[start_cols:-end_cols, start_rows:-end_rows]
        warped_frame = cv2.resize(warped_frame, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)

        write_frame_to_video(vw, warped_frame, height=orig_height, width=orig_width)
        prev_frame = frame

    vw.release()
    vc.release()
    cv2.destroyAllWindows()
    
