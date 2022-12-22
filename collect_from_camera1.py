import cv2
import face_detection
from pathlib import Path
from datetime import datetime
from insightface.src.common import face_preprocess
import numpy as np
import time
import os


class facedetector():
    def __init__(self,args):
        self.detector = face_detection.build_detector(
        'RetinaNetMobileNetV1', confidence_threshold=.5, nms_iou_threshold=.3)
        self.args=args
        print("[INFO] Face Detection Model Loaded")
    def face_collection(self):
        #Path(self.args["output"]).mkdir(parents=True,exist_ok=True)
        dataset=Path("dataset").glob("*/*")
        for i in dataset:
                print(i)
                Path(f"datasets/train/{i.parent.stem}").mkdir(parents=True,exist_ok=True)
                frame=cv2.imread(str(i))
                #cap=cv2.VideoCapture(0)
                num_images=self.args["faces"]
                initial=0
                #while cap.isOpened() and initial<=num_images:
                #ret, frame = cap.read()
                print(frame.shape)
                frame=frame[:, :, ::-1]
                print(frame.shape)
                start_time=time.time()
                detections = self.detector.detect(frame)
                frame=frame[:, :, ::-1]
                end_time=time.time()
                if len(detections[0]) != 0:
                        for bbox in detections:
                            bbox_xmin = bbox[0]
                            bbox_ymin = bbox[1]
                            bbox_xmax = bbox[2]
                            bbox_ymax = bbox[3]
                            dtString = str(datetime.now().microsecond)
                            #frame=np.array(frame)
                            crop = frame[int(bbox_ymin):int(bbox_ymax), int(bbox_xmin):int(bbox_xmax)]
                            #max_bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                            #nimg = face_preprocess.preprocess(cv2.imread(crop), max_bbox, landmarks=None, image_size='112,112')
                            cv2.imwrite(os.path.join("datasets/train/"+i.parent.stem+"/"+"{}.jpg".format(dtString)),crop)
                            cv2.rectangle(frame, (int(bbox_xmin), int(bbox_ymin)), (int(bbox_xmax), int(bbox_ymax)), (0, 0, 255), 2)
                            print("[INFO] {} Image Captured".format(initial + 1),"fps :",(end_time-start_time))
                            initial += 1
                #cv2.imshow("Face detection", frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #break

            #cap.release()
            #cv2.destroyAllWindows()
