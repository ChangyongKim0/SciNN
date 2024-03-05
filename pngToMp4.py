import os
import cv2


def pngToMp4(png_file_path, mp4_file_name, fps=60):
    paths = os.listdir(png_file_path)
    img0 = cv2.imread(f"{png_file_path}/{paths[0]}")
    height, width, _ = img0.shape
    size = (width, height)
    out = cv2.VideoWriter(
        f"{mp4_file_name}.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for path in paths:
        img = cv2.imread(f"{png_file_path}/{path}")
        out.write(img)
    out.release()
