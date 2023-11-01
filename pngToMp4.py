import os
import cv2


def pngToMp4(png_file_path, mp4_file_name, fps=60):
    frame_array = []
    paths = os.listdir(png_file_path)
    for path in paths:
        img = cv2.imread(f"{png_file_path}/{path}")
        height, width, _ = img.shape
        size = (width, height)
        frame_array.append(img)
    out = cv2.VideoWriter(
        f"{mp4_file_name}.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for frame in frame_array:
        out.write(frame)
    out.release()
