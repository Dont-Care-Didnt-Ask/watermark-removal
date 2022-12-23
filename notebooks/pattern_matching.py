from skimage.util import img_as_float64

import numpy as np
import cv2 as cv
import skimage
import scipy

def create_mask(image, template, threshold):
    matched = skimage.feature.match_template(image, template, pad_input=True)
    matched[matched < matched.max() * threshold] = 0
    matched = matched != 0    
    return np.float64(matched)

def center_pipeline(image, templates, thresh1=0.7, thresh2=0.6, dilate=0):
    final_mask = create_mask(image, templates, thresh1)
    final_mask[final_mask > thresh2] = 1

    
    for i in range(dilate):
        final_mask = skimage.morphology.dilation(final_mask)

    final_mask = scipy.ndimage.distance_transform_edt(final_mask)

    return final_mask

def dist(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def filter_coords(coords):
    filtered = []
    for peak in coords:
        subpeaks = coords.copy()
        subpeaks.remove(peak)
        if all(dist(peak, other) > 10 for other in subpeaks):
            filtered.append(peak)

    return filtered

def find_coords(img, tpl):
    res_img = np.zeros_like(img[..., 0])
    res0 = center_pipeline(skimage.color.rgb2lab(img)[..., 0], skimage.color.rgb2lab(tpl)[..., 0], thresh1=0.65, thresh2=0.62, dilate=0)
    res1 = center_pipeline(skimage.color.rgb2lab(img)[..., 1], skimage.color.rgb2lab(tpl)[..., 1], thresh1=0.65, thresh2=0.62, dilate=0)
    res2 = center_pipeline(skimage.color.rgb2lab(img)[..., 2], skimage.color.rgb2lab(tpl)[..., 2], thresh1=0.65, thresh2=0.62, dilate=0)
    res = res0 + res1 + res2
    res[res != 0] = 1

    peaks = [(x, y) for x, y in skimage.feature.peak_local_max(res, 10)]

    for x, y in peaks:
        res_img[int(x)-tpl.shape[0]//2:int(x)+tpl.shape[0]//2 - 10, int(y)-tpl.shape[1]//2:int(y)+tpl.shape[1]//2 - 10] = 1

    labeled = skimage.measure.label(skimage.morphology.convex_hull_object(res_img))
    res_coords = []
    for i in np.unique(labeled):
        if i != 0:
            mask = labeled.copy()
            mask[mask != i] = 0
            mask[mask == i] = 1

            result = np.where(mask == np.amax(mask))
            x1 = np.min(result[0])
            y1 = np.min(result[1])

            res_coords.append((y1 / img.shape[1], x1 / img.shape[0], tpl.shape[1] / img.shape[1], tpl.shape[0] / img.shape[0]))

    return res_coords

def load_preprocess_image(img):
    img = img_as_float64(img)
    img = img / img.max()

    img = img[:, :, ::-1]

    return img

def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    img = load_preprocess_image(img)
    query = load_preprocess_image(query)
    list_of_bboxes = find_coords(img, query)
    return list_of_bboxes

if __name__ == "__main__":
    img = cv.imread("/home/chameleon/University/Term2/I2CV/HW3/train/train_0.jpg")
    query = cv.imread("/home/chameleon/University/Term2/I2CV/HW3/train/template_0_0.jpg")
    
    print(predict_image(img, query))