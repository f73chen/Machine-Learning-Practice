import cv2
import os

def make_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('qlearn.avi', fourcc, 60.0, (1200, 900))
    
    for i in range (0, 20000, 100):
        imgPath = f"qtable_charts\{i}.png"
        print(imgPath)
        frame = cv2.imread(imgPath)
        out.write(frame)
        
    out.release()
    
make_video()