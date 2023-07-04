import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import cv2
from scipy.ndimage import filters

def compute_harris_response(image: np.ndarray, 
                            alpha: float = 0.04, 
                            window_size: int = 5,
                            thresh_percent: float = 0.1, 
                            ) -> np.ndarray:
    I_x, I_y = np.gradient(image.astype(np.float32))

    I_xx = I_x * I_x
    I_xy = I_x * I_y
    I_yy = I_y * I_y

    g = np.ones((window_size,window_size))
    S_xx = filters.convolve(I_xx, g)
    S_xy = filters.convolve(I_xy, g)
    S_yy = filters.convolve(I_yy, g)

    R = (S_xx*S_yy - S_xy*S_xy) - alpha*(S_xx + S_yy)**2

    threshold = thresh_percent*R.max()
    R[R < threshold] = 0
    R[R > 0] = 1
    R = R.astype(np.uint8)

    return R
    
image = cv2.imread('R.jpeg')
I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def try_load_image():
    global image, I
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()

    if path == '':
        return
    
    image = cv2.imread(path)
    if image is None:
        return
    
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    update(0, image, update_view=True)

fig, ax = plt.subplots(1, 1)

alpha_slider = widgets.Slider(ax=plt.axes([0.1, 0.1, 0.8, 0.03]), label='Alpha', valmin=0.01, valmax=0.07, valinit=0.04, valstep=0.0001)
thresh_slider = widgets.Slider(ax=plt.axes([0.1, 0.15, 0.8, 0.03]), label='Threshold', valmin=0, valmax=0.1, valinit=0.01, valstep=0.0001)
window_slider = widgets.Slider(ax=plt.axes([0.1, 0.2, 0.8, 0.03]), label='Window Size', valmin=3, valmax=15, valinit=5, valstep=2)
corners_checkbox = widgets.CheckButtons(ax=plt.axes([0.1, 0.05, 0.1, 0.05]), labels=['Show Corners'], actives=[True])
image_button = widgets.Button(ax=plt.axes([0.2, 0.05, 0.1, 0.05]), label='Load Image')
image_button.on_clicked(lambda _: try_load_image())

ax.set_position([0.1, 0.25, 0.8, 0.75])
elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

def colorize_mask(image, mask, color):
    return image * (1-mask[:, :, np.newaxis]) + mask[:, :, np.newaxis] * color

def update(val, image, update_view=False):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    im_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    alpha = alpha_slider.val
    thresh_percent = thresh_slider.val
    window_size = int(window_slider.val)
    show_corners = corners_checkbox.get_status()[0]
    
    if show_corners:
        R = compute_harris_response(image=I, 
                                            alpha=alpha, 
                                            thresh_percent=thresh_percent,
                                            window_size=window_size)
        ax.clear()

        R = cv2.dilate(R, elem, iterations=1)

        im_copy = colorize_mask(im_copy, R, np.array([255, 0, 0])).astype(np.uint8)

    ax.imshow(im_copy)
    if not update_view:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    fig.canvas.draw_idle()

window_slider.on_changed(lambda val: update(val, image))
alpha_slider.on_changed(lambda val: update(val, image))
thresh_slider.on_changed(lambda val: update(val, image))
corners_checkbox.on_clicked(lambda val: update(val, image))

ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
update(0, image)

plt.show()