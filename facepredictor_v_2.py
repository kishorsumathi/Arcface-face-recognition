import face_detection
from insightface.deploy import face_model
import warnings
import sys
import dlib
from insightface.src.common import face_preprocess
# from src.insightface.deploy import face_model
import torch
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cosine
from numpy.linalg import norm
sys.path.append('insightface/deploy')
sys.path.append('insightface/src/common')
import tensorflow 
from tensorflow import keras
from keras.models import load_model
import numpy as np
import time
import pickle
import cv2


class FacePredictor():
    def __init__(self):
        try:
            self.image_size = '112,112'
            self.model = "insightface/models/model-y1-test2/model,0"
            self.det = 0
            self.detector = face_detection.build_detector(
  'RetinaNetMobileNetV1', confidence_threshold=.5, nms_iou_threshold=.3)
            self.embedding_model = face_model.FaceModel(self.image_size, self.model, self.det)

            self.embeddings = "faceEmbeddingModels/embeddings.pickle"
            self.le = "faceEmbeddingModels/le.pickle"
            self.data = pickle.loads(open(self.embeddings, "rb").read())
            self.le = pickle.loads(open(self.le, "rb").read())

            self.embeddings = np.array(self.data['embeddings'])
            self.labels = self.le.fit_transform(self.data['names'])

            # Load the classifier model
            self.model = load_model("faceEmbeddingModels/my_model.h5")


        except Exception as e:
            print(e)

    # Define distance function
    @staticmethod
    def findCosineDistance(vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1-(a / (np.sqrt(b) * np.sqrt(c)))


    @staticmethod
    def CosineSimilarity(test_vec, source_vecs):
        """
        Verify the similarity of one vector to group vectors of one class
        """
        cos_dist = 0
        
        for source_vec in source_vecs:
            cos_dist+=FacePredictor.findCosineDistance(test_vec,source_vec)
        return cos_dist / len(source_vecs)
    
    @staticmethod
    def on_mouse(event, x, y, flags, userdata):
        global state, p1, p2
        if event == cv2.EVENT_LBUTTONUP:
            if state == 0:
                p1 = (x,y)
                state += 1
            elif state == 1:
                p2 = (x,y)
                state += 1

    def detectFace(self):
        # Initialize some useful arguments
        cosine_threshold = 0.4
        proba_threshold = 0.4
        comparing_num =8
        global state, p1, p2
        frames = 0
        trackers = []
        texts = []
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        p1, p2 = None, None
        x_1,y_1=0,0
        w_1,z_1=0,0
        state = 0
        cv2.setMouseCallback('Frame', self.on_mouse)
        while True:
                start=time.time()
                ret, frame = cap.read()
                if p1!=None and p2!=None:
                    if ret:
                        x_1,y_1=p1
                        w_1,z_1=p2
                        frame_crop = frame[y_1:z_1,x_1:w_1]
                if p1==None or p2==None:
                    frame_crop=frame
                if state > 1:
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 5)
                if ret:
                    frames += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_crop=frame_crop[:, :, ::-1]
                    bboxes =self.detector.detect(frame_crop)
                    frame_crop=frame_crop[:, :, ::-1]
                    if frames % 3 == 0:
                        trackers = []
                        texts = []
                        if len(bboxes) != 0:
                            for bboxe in bboxes:
                                bbox = bboxe
                                bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
                                bbox=bbox.astype("int32")
                                crop = frame_crop[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                                if bboxe[4]>0.7 and len(crop)!=0 :
                                    crop = face_preprocess.preprocess(crop, image_size='112,112')
                                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    crop = np.transpose(crop, (2, 0, 1))
                                    embedding = self.embedding_model.get_feature(crop).reshape(1, -1)
                                    text = "Unknown"
                                    preds =  self.model.predict(embedding)
                                    preds = preds.flatten()
                                    j = np.argmax(preds)
                                    proba = preds[j]
                                    match_class_idx = ( self.labels == j)
                                    match_class_idx = np.where(match_class_idx)[0]
                                    selected_idx = np.random.choice(match_class_idx, comparing_num)
                                    compare_embeddings = self.embeddings[selected_idx]
                                    cos_similarity =  self.CosineSimilarity(embedding, compare_embeddings)
                                    print(cos_similarity)
                                    if 1-cos_similarity>cosine_threshold and proba > proba_threshold:
                                        name =  self.le.classes_[j]
                                        text = "{}".format(name)
                                        print("Recognized: {} <{:.2f}>".format(name, proba * 100))
                                    else:
                                        print(f"Recognized:{text}")
                                    y = ((bbox[1]+y_1) - 10) if ((bbox[1]+y_1) - 10) > 10 else ((bbox[1]+y_1) + 10)
                                    cv2.putText(frame, text, (bbox[0]+x_1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)
                                    cv2.rectangle(frame, (bbox[0]+x_1, bbox[1]+y_1), (bbox[2]+x_1, bbox[3]+y_1), (179, 0, 149), 4)
                                    tracker = dlib.correlation_tracker()
                                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                                    tracker.start_track(frame, rect)
                                    trackers.append(tracker)
                                    texts.append(text)
                                else:
                                    pass
                    else:
                        for tracker, text in zip(trackers, texts):
                            tracker.update(rgb)
                            pos = tracker.get_position()

                            # unpack the position object
                            startX = int(pos.left())+x_1
                            startY = int(pos.top())+y_1
                            endX = int(pos.right())+x_1
                            endY = int(pos.bottom())+y_1

                            cv2.rectangle(frame, (startX, startY), (endX, endY), (179, 0, 149), 4)
                            cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)
                    end=time.time()
                    print("[INFO] Total time taken for inferencing :",end-start)
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

