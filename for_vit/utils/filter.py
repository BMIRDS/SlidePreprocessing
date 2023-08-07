import numpy as np
from skimage.util.shape import view_as_windows
from skimage.morphology import \
    (remove_small_objects, remove_small_holes, binary_erosion, binary_dilation)
import cv2


SIZE = 1

def is_purple_dot(r, g, b):
    rb_avg = (r + b) / 2
    if r > g - 10 and b > g - 10 and rb_avg > g + 20:
        return True
    return False

def is_purple(crop):
    crop = crop.reshape(SIZE, SIZE, 3)
    for x in range(crop.shape[0]):
        for y in range(crop.shape[1]):
            r = crop[x, y, 0]
            g = crop[x, y, 1]
            b = crop[x, y, 2]
            if is_purple_dot(r, g, b):
                return 1
    return 0


def filter_purple(img):
    h, w, d = img.shape
    step = SIZE
    img_padding = np.zeros((h + step - 1, w + step - 1, d))
    img_padding[:h, :w, :d] = img
    img_scaled = view_as_windows(img_padding, (SIZE, SIZE, 3), 1)
    return np.apply_along_axis(is_purple, -1,
                               img_scaled.reshape(h, w, -1)).astype(int)

def is_stained_dot(r, g, b):
    rb_avg = (r + b) / 2
    if r > g - 10 and b > g - 10 and rb_avg > g + 20:
        return True
    return False

def is_stained(crop):
    crop = crop.reshape(SIZE, SIZE, 3)
    for x in range(crop.shape[0]):
        for y in range(crop.shape[1]):
            r = crop[x, y, 0]
            g = crop[x, y, 1]
            b = crop[x, y, 2]
            if is_stained_dot(r, g, b):
                return 1
    return 0

def filter_stained(img):
    # TODO: Commit filter changes
    h, w, d = img.shape
    step = SIZE
    img_padding = np.zeros((h + step - 1, w + step - 1, d))
    img_padding[:h, :w, :d] = img
    img_scaled = view_as_windows(img_padding, (SIZE, SIZE, 3), 1)
    return np.apply_along_axis(is_stained, -1,
                               img_scaled.reshape(h, w, -1)).astype(int)


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
  Create a mask to filter out pixels where the red, green, and blue channel values are similar.
  Args:
    np_img: RGB image as a NumPy array.
    tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
  """
    (h, w, c) = rgb.shape

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)
    return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh):
    """
    credit: https://github.com/CODAIT/deep-histopath/blob/c8baf8d47b6c08c0f6c7b1fb6d5dd6b77e711c33/deephistopath/wsi/filter.py#L771
    """
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return result


def filter_blue_pen(rgb):
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
           filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
           filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
           filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
           filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
           filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
           filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
           filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
           filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
           filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
           filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh):
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    return result


def filter_green_pen(rgb):
    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
           filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
           filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
           filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
           filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
           filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
           filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
           filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
           filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
           filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
           filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
           filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
           filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)
    return result



def select_mask_instances(mask):
    # contours # if you're using OpenCV 3* then it returns as _, contours, _
    contours, _ = cv2.findContours(mask.astype(np.uint8)*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    threshold = sum([cv2.contourArea(con) for con in contours]) * 0.8

    final_mask = np.zeros_like(mask).astype(float)
    total_area = 0
    for con in contours:
        area = cv2.contourArea(con)
        mask_i = np.zeros_like(mask).astype(float)
        cv2.drawContours(mask_i, [con], -1, (100, 150, 0), -1)
        final_mask += (mask_i > 0).astype(float)
        total_area += area
        if total_area > threshold:
            break
    return final_mask > 0


def filter_composite(imgs, style):
    # select the region with colors
        
    if style == 'gram_stains':
        mask_s = np.zeros_like(imgs[0][:, :, 0]).astype(int)
        for i, img in enumerate(imgs):
            mask_s += filter_stained(img)

        mask_s = mask_s > 0
        mask = mask_s 
        
    else:
        if style != 'default':
            print("[WARNING] Unrecognized filtering style, proceeding with default settings")
        mask_s = np.zeros_like(imgs[0][:, :, 0]).astype(int)
        for i, img in enumerate(imgs):
            mask_s += filter_purple(img)

        mask_s = mask_s > 0
        # mask_s = binary_erosion(binary_dilation(mask_s, np.ones((10,10))), np.ones((10,10)))
        # filter pen marks
        mask_no_pen = filter_blue_pen(imgs[0]) & filter_green_pen(imgs[0])
        mask = mask_s & mask_no_pen
        mask = remove_small_objects(mask > 0, 400)
        # mask = select_mask_instances(mask)

    return mask
