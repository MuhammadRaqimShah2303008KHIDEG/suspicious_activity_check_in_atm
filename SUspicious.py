import cv2
from ultralytics import YOLO
from collections import defaultdict
import csv
from datetime import datetime

def search_suspicious(model1, model2, video_path):
    model_1 = model1
    model_2 = model2
    # Load video
    video_path = video_path
    cap = cv2.VideoCapture(video_path)
    count = 0
    card_detected = False

    # Get frames per second of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Resize frame for faster processing (adjust resolution as needed)
            frame = cv2.resize(frame, (1300, 800))
            results = model_1.track(source=frame, show=False, project='./result', tracker="bytetrack.yaml", conf=0.4)
            for result in results:
                class_name = result.boxes.cls.tolist()
                conf_name = result.boxes.conf.tolist()
                res = dict(zip(class_name, conf_name))
            # res = [{cls: conf} for result in results for cls, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())]
                print(res)
                if len(res) != 0:
                    results2 = model_2.track(source=frame, show=True, project='./result', tracker="bytetrack.yaml", conf=0.4)
                    for result2 in results2:
                        class_name2 = result2.boxes.cls.tolist()
                        conf_name2 = result2.boxes.conf.tolist()
                        res2 = dict(zip(class_name2, conf_name2))
                        print(res2)
                        if 0.0 in class_name2 and res2[0.0] >= 0.7:
                            print("ATM Card is Detected")
                            count = 0
                            # return "ATM Card is Detected"
                        elif count >= 100:
                            count = 0
                            print('Guard is not Scanning')
                        else:
                            print(count)
                            count +=1    
                else:
                    pass
        else:
            print(f"Working with frame empty frame, no frame detected")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model1 = YOLO('track.pt')
    model2 = YOLO('ATM_Card.pt')
    # Load video
    video_path = 'test2.mp4'
    search_suspicious(model1, model2, video_path)