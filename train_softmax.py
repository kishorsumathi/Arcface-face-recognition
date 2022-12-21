from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# from keras.models import load_model
import matplotlib.pyplot as plt
# from softmax import SoftMax
import numpy as np
import pandas as pd
import argparse
import pickle
import time
from sklearn.metrics import precision_recall_curve
from training import SoftMax



class TrainFaceRecogModel:

    def __init__(self, args):

        self.args = args
        self.data = pickle.loads(open(args["embeddings"], "rb").read())

    def trainKerasModelForFaceRecognition(self):
        # Encode the labels
        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"])
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder()
        labels = one_hot_encoder.fit_transform(labels).toarray()

        embeddings = np.array(self.data["embeddings"])

        # Initialize Softmax training model arguments
        BATCH_SIZE = 32
        EPOCHS = 50
        input_shape = embeddings.shape[1]

        # Build sofmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # Create KFold
        start=time.time()
        cv = KFold(n_splits = 2, random_state = 42, shuffle=True)
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': [], 'epoch': [],'precision':[],"recall":[],"val_precision":[],"val_recall":[]}

        # Train
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
            his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))
            print(his.history['accuracy'])

            history['acc'] += his.history['accuracy']
            history['val_acc'] += his.history['val_accuracy']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']
            history['precision'] +=his.history["precision"]
            history['recall'] +=his.history["recall"]
            history['val_precision'] +=his.history["val_precision"]
            history['val_recall'] +=his.history["val_recall"]
            model_history = pd.DataFrame(his.history)
            print(model_history)
            model_history['epoch'] =model_history.index
            history['epoch'] +=model_history['epoch'].tolist()
            # for i in ["acc","loss","precision","recall"] :
            #     plt.plot(history['epoch'],history[i], "-b", label=i,linewidth=2)
            #     plt.plot(history['epoch'],history['val_{i1}'.format(i1=i)], "-r", label='val_{i1}'.format(i1=i),linewidth=2)
            #     plt.legend(loc="upper left")
            #     if i!="loss":

            #         plt.ylim(0.0, 1)
            #         plt.xlim(0,EPOCHS)
            #         plt.title('Model_{i1}'.format(i1=i))
            #         plt.savefig("img_graphs/{i1}".format(i1=i))
            #         plt.close()
            #     else:
            #         plt.ylim(0.0,1)
            #         plt.xlim(0,EPOCHS)
            #         plt.title('Model_{i1}'.format(i1=i))
            #         plt.savefig("img_graphs/{i1}".format(i1=i))
            #         plt.close()

            # # write the face recognition model to output
            # end=time.time()
            # print(f"total time taken: {end-start}")
            # model.save(self.args['model'])
            # f = open(self.args["le"], "wb")
            # f.write(pickle.dumps(le))
            # f.close()
