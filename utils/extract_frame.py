import cv2
from PIL import Image
import numpy as np

cap = cv2.VideoCapture("D:/mouse/mouse_30fps/videos/K-2.mp4")  
isOpened = cap.isOpened  
fps = cap.get(cv2.CAP_PROP_FPS)

imageNum = 2790 - 1
sum=0
timef=1  

while (isOpened):

    sum+=1

    (frameState, frame) = cap.read()  

    if frameState == True and (sum % timef==0) and (2790 <= sum < 4590):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))

        frame = np.array(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        imageNum = imageNum + 1
        frameindex = imageNum * 1
        fileName = f'D:/mouse/mouse_30fps_consecutive/frames/K-2/img{frameindex:05d}.png' 
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(fileName + " successfully write in") 

    elif frameState == False:
        break

print('finish')
cap.release()