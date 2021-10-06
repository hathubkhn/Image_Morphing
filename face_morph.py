import sys
import cv2
import sys
import numpy as np
import argparse
from scipy.spatial import Delaunay
import imageio
from tqdm import tqdm
import dlib
from skimage import io

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def extract_features(img_path):
    img = io.imread(img_path)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)

    vec = [(0, 0), (0, img.shape[0]-1)]

    for j in range(0, 68):
        vec.append((shape.part(j).x, shape.part(j).y))
    vec.append((img.shape[1]-1, 0)) 
    vec.append((img.shape[1]-1, img.shape[0]-1))

    return vec

def apply_affine_transform(src, src_tri, target_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(target_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(0, 3): #3 dimensions (x, y, z), t1 = [src[x], src[y], src[z]]
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1]))) #morph
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1]))) #src
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1]))) #target

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0) #making morph image

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3]) #size of morph

    # Determine warp triangles
    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask


def get_morph(alpha, src_img, src_points, target_img, target_points, del_triangles):
    weighted_pts = []
    # Step1: Find location of feature points in morphed image (x, y)
    for i in range(0, len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * target_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * target_points[i][1]
        weighted_pts.append((x, y)) #points in morphed image

    img_morph = np.zeros(src_img.shape, dtype=src_img.dtype)
    for i in range(del_triangles.shape[0]):
        x, y, z = del_triangles[i]
        t1 = [src_points[x], src_points[y], src_points[z]] #src

        t2 = [target_points[x], target_points[y], target_points[z]] #target
        t = [weighted_pts[x], weighted_pts[y], weighted_pts[z]] #morphed
        morph_triangle(src_img, target_img, img_morph, t1, t2, t, alpha) 

    return cv2.cvtColor(np.uint8(img_morph), cv2.COLOR_RGB2BGR)

def writeFaceLandmarksToLocalFile(faceLandmarks, fileName):
  with open(fileName, 'w') as f:
    for p in faceLandmarks:
      f.write("%s %s\n" %(int(p[0]),int(p[1])))

  f.close()

def main():
    #from vid_lib import Video
    parse = argparse.ArgumentParser()
    parse.add_argument('-i1', '--image1', type = str)
    parse.add_argument('-i2', '--image2', type = str)
    parse.add_argument('-o', '--output', type = str)
    args = parse.parse_args()
    
    SRC_IMG = args.image1  
    TARGET_IMG = args.image2 
    VID_FILE = args.output
    src_img = cv2.imread(SRC_IMG)
    target_img = cv2.imread(TARGET_IMG)
    src_points = extract_features(SRC_IMG) #72 points
    target_points = extract_features(TARGET_IMG) #72 points
    
    avg_points = []
    for i in range(0, len(src_points)):
        x = 0.5 * src_points[i][0] + 0.5 * target_points[i][0]
        y = 0.5 * src_points[i][1] + 0.5 * target_points[i][1]
        avg_points.append((int(x), int(y)))
    
    # Define del_triangles points
    del_triangles = Delaunay(avg_points)
    del_triangles = del_triangles.simplices
    
    STEPS = 300
    FPS = 30
    FREEZE_STEPS = 10
    
    video = imageio.get_writer('video.mp4', mode = 'I', fps = FPS, codec = 'libx264', bitrate = '16M')
    for j in tqdm(range(STEPS)):
        repeat = FREEZE_STEPS if j==0 or j==(STEPS -1) else 1
        for i in range(repeat):
            video.append_data(get_morph(j/STEPS, src_img, src_points, target_img, target_points, del_triangles))
    video.close()

if __name__ == '__main__':
    main()
