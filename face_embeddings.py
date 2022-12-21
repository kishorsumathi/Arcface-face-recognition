import sys

# from insightface.deploy import face_model
# from insightface.deploy import face_model
# from src.insightface.deploy import face_model
from insightface.deploy import face_model
from pathlib import Path

sys.path.append('/insightface/deploy')
sys.path.append('/insightface/src/common')

from imutils import paths
import time
import numpy as np
# import face_model
import pickle
import cv2
import os


class GenerateFaceEmbedding:

    def __init__(self, args):
        self.args = args
        self.image_size = '112,112'
        self.model = "./insightface/models/model-y1-test2/model,0"
        #self.threshold = 1.24
        self.det = 0

    def genFaceEmbedding(self):
        # Grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(self.args.dataset))

        # Initialize the faces embedder
        embedding_model = face_model.FaceModel(self.image_size, self.model,self.det)

        # Initialize our lists of extracted facial embeddings and corresponding people names
        knownEmbeddings = []
        knownNames = []

        # Initialize the total number of faces processed
        total = 0

        # Loop over the imagePaths
        start=time.time()
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            class_name = imagePath.split(os.path.sep)[-2]
            class_name=Path(class_name).name

            # load the image
            image = cv2.imread(imagePath)
            image=cv2.resize(image,(112,112))
            # convert face to RGB color
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1)) ##chw
            # Get the face embedding vector
            face_embedding = embedding_model.get_feature(nimg)

            # add the name of the person + corresponding face
            # embedding to their respective list
            knownNames.append(class_name)
            knownEmbeddings.append(face_embedding)
            total += 1

        print(total, " faces embedded")
        end=time.time()
        print(f"total_time_taken:{end-start}")

        # save to output
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(self.args.embeddings, "wb")
        f.write(pickle.dumps(data))
        f.close()
        return 0
