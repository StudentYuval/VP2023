import argparse
import cv2
import numpy as np

def get_sorted_image_path_list(input_dir):
    import os
    image_list = os.listdir(input_dir)
    image_list.sort(key=lambda s: int(s.split('_')[1].split('.')[0]))
    image_list = [os.path.join(input_dir, image) for image in image_list]
    return image_list

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-dir', type=str, default='finished_run/video_debug')
parser.add_argument('-o', '--output-file', type=str, default='output.avi')
parser.add_argument('-t', '--is-trim-frames', action='store_true', default=False)
parser.add_argument('-sr', '--start-row', type=int, default=10)
parser.add_argument('-er', '--end-row', type=int, default=30)
parser.add_argument('-sc', '--start-colomn', type=int, default=2)
parser.add_argument('-ec', '--end-colomn', type=int, default=30)
args = parser.parse_args()

images_path_list = get_sorted_image_path_list(args.input_dir)
first_image = cv2.imread(images_path_list[0])

vw = cv2.VideoWriter(args.output_file, 
                     fourcc=cv2.VideoWriter_fourcc(*'XVID'),
                     fps=30,
                     frameSize=(331, 270),
                     isColor=False)

for path in images_path_list:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    if args.is_trim_frames:
        img = cv2.resize(img, (331, 270), interpolation=cv2.INTER_AREA)
        img = img[args.start_row:-args.end_row, args.start_colomn:-args.end_colomn]
    img = cv2.resize(img, (331, 270), interpolation=cv2.INTER_AREA)
    vw.write(img.astype(np.uint8))

vw.release()
