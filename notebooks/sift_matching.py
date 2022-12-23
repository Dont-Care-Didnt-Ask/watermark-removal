import cv2
import numpy as np
from skimage.util import img_as_float, img_as_ubyte
from skimage.morphology import convex_hull_object, erosion, disk

def build_mask(kp1, kp2, matches, temp_shape, img_shape, min_match_count=10):
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            good.append(m)

    img_mask = None
    dst = None
    
    if len(good) >= min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = temp_shape
        pts = np.float32([[0, 0],[0, h - 1],[w - 1, h - 1],[w - 1, 0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        img_mask = cv2.polylines(np.zeros(img_shape), [np.int32(dst)], True, 255, 1, cv2.LINE_AA)
        img_mask = convex_hull_object(img_mask)
        img_mask = erosion(img_mask, footprint=disk(3))
    else:
        print( f"Not enough matches are found - {len(good)}/{min_match_count}")

    return img_mask, dst


def get_bbox(dst, img_shape):
    x, y = dst[0, 0]
    bbox_w = np.abs(x - dst[2, 0, 0])
    bbox_h = np.abs(y - dst[1, 0, 1])
    
    h, w = img_shape
    return [x / w, y / h, bbox_w / w, bbox_h / h]


def mask_from_bbox(bbox, img_shape):
    background = np.zeros(img_shape)
    x, y, bbox_w, bbox_h = bbox
    h, w = img_shape
    
    x_min = int(x * w)
    y_min = int(y * h)
    x_max = int((x + bbox_w) * w)
    y_max = int((y + bbox_h) * h)
    
    background[y_min:y_max, x_min:x_max] = 1
    
    return background


def iou(mask1, mask2):
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    if (mask1 | mask2).sum() < 1 / mask1.size:
        return 0
    
    return (mask1 & mask2).sum() / (mask1 | mask2).sum()


def predict_image(img, query):
    max_iter = 30
    # BGR -> RGB
    img = cv2.resize(img_as_float(img[..., ::-1]), (img.shape[1] * 600 // img.shape[0], 600))
    temp = cv2.resize(img_as_float(query[..., ::-1]), (query.shape[1] * 140 // query.shape[0], 140))
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_as_ubyte(temp), None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=20)
    search_params = dict(checks=20)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    answer = []

    for step in range(max_iter):
        kp2, des2 = sift.detectAndCompute(img_as_ubyte(img), None)
        matches = flann.knnMatch(des1, des2, k=2)
        mask, dst = build_mask(kp1, kp2, matches, temp.shape[:2], img.shape[:2])

        if mask is None:
            break

        bbox = get_bbox(dst, img.shape[:2])

        rect = mask_from_bbox(bbox, img.shape[:2])

        i = iou(rect, mask)
        if i > 0.80:
            answer.append(bbox)
            img = np.maximum(img, rect[..., None]) 
        
    return answer