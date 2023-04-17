import cv2
import glob
import natsort

images = (glob.glob('./heatmap/*.png'))

#  정렬 작업
images = natsort.natsorted(images)

pathOut = './heatmapVideo/output1-.mp4'
fps = 11.5

frame_array = []

for idx, path in enumerate(images):
    if (idx % 2 == 0) | (idx % 5 == 0):
        continue
    img = cv2.imread(path)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for i in range(len(frame_array)):
    # writing to an image array
    out.write(frame_array[i])

out.release()