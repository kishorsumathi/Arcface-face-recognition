import face_detection
from insightface.deploy import face_model
import warnings
import sys
import dlib
# from src.insightface.deploy import face_model
import torch
warnings.filterwarnings('ignore')
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
            print(self.labels)

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
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def CosineSimilarity(test_vec, source_vecs,labels):
        """
        Verify the similarity of one vector to group vectors of one class
        """
        dist=[]
        cos_dist = 0
        for i in range(len(np.unique(labels))):
            match_class_idx = (labels == i)
            match_class_idx = np.where(match_class_idx)[0]
            selected_idx = np.random.choice(match_class_idx,8)
            compare_embeddings = source_vecs[selected_idx]
            cos_dist = 0
            for source_vec in compare_embeddings:
                cos_dist += FacePredictor.findCosineDistance(test_vec, source_vec)
            dist.append(1-(cos_dist / len(compare_embeddings )))
        index=np.argmax(dist)
        #print(dist)
        #print(index)
        return index,dist[index]

    def detectFace(self):
        # Initialize some useful arguments
        cosine_threshold = 0.4
        proba_threshold = 0.6
        comparing_num =8
        frames = 0
        trackers = []
        texts = []
        cap = cv2.VideoCapture(0)
        while True:
                start=time.time()
                ret, frame = cap.read()
                if ret:
                    frames += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame=frame[:, :, ::-1]
                    bboxes =self.detector.detect(frame)
                    frame=frame[:, :, ::-1]
                    if frames % 3 == 0:
                        trackers = []
                        texts = []
                        if len(bboxes) != 0:
                            for bboxe in bboxes:
                                bbox = bboxe
                                bbox = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
                                bbox=bbox.astype("int32")
                                crop = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
                                if len(crop)!=0 and bboxe[4]>0.7:
                                    crop=cv2.resize(crop,(112,112))
                                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    crop = np.transpose(crop, (2, 0, 1))
                                    embedding = self.embedding_model.get_feature(crop).reshape(1, -1)
                                    text = "Unknown"
                                    #match_class_idx = ( self.labels == j)
                                    #match_class_idx = np.where(match_class_idx)[0]
                                    #selected_idx = np.random.choice(match_, comparing_num)
                                    #compare_embeddings = self.embeddings[selected_idx]
                                    j,cos_similarity =  self.CosineSimilarity(embedding,self.embeddings,self.labels)
                                    #print("probablity of model : ",cos_similarity)
                                    print("similarity :" ,cos_similarity)
                                    if cos_similarity>cosine_threshold:
                                            name = self.le.classes_[j]
                                            text = "{}".format(name)
                                            print("Recognized: {} <{:.2f}>".format(name, cos_similarity* 100))
                                    else:
                                            print(f"Recognized:{text}")
                                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                                    cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)
                                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (179, 0, 149), 4)
                                    tracker = dlib.correlation_tracker()
                                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                                    tracker.start_track(rgb, rect)
                                    trackers.append(tracker)
                                    texts.append(text)
                                else:
                                    pass
                    else:
                        for tracker, text in zip(trackers, texts):
                            tracker.update(rgb)
                            pos = tracker.get_position()

                            # unpack the position object
                            startX = int(pos.left())
                            startY = int(pos.top())
                            endX = int(pos.right())
                            endY = int(pos.bottom())

                            cv2.rectangle(frame, (startX, startY), (endX, endY), (179, 0, 149), 4)
                            cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)
                end=time.time()
                print("[INFO] Total time taken for inferencing :",end-start)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

