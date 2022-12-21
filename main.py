import argparse
from collect_from_camera import facedetector
from face_embeddings import GenerateFaceEmbedding
from train_softmax import TrainFaceRecogModel
from facepredictor import FacePredictor


def collect_images():
        name= str(input("Enter your name are the person name to be detected : "))
        ap = argparse.ArgumentParser()
        ap.add_argument("--faces", default=20,
                        help="Number of faces that camera will get")
        ap.add_argument("--output", default="datasets/train/" + name,
                        help="Path to faces output")
        
        ap.add_argument("--training", required=True,help="True if training required or False for inferencing")

        args = vars(ap.parse_args())

        if args["training"]=="True":
                trnngDataCollctrObj = facedetector(args)
                trnngDataCollctrObj.face_collection()
                print(f"[INFO] {args['output']} Face images collected")
                return  args["training"]
        else:
                print("[INFO] inference Process Loading....,")
                return args["training"]
def getFaceEmbedding():

        ap = argparse.ArgumentParser()

        ap.add_argument("--dataset", default="datasets/train/",
                        help="Path to training dataset")
        ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle")
        ap.add_argument('--image-size', default='112,112', help='')
        ap.add_argument('--model', default='insightface/models/model-y1-test2/model,0', help='path to load model.')
        ap.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        ap.add_argument("--training", required=True,help="True if training required or False for inferencing")
        args = ap.parse_args()

        genFaceEmbdng = GenerateFaceEmbedding(args)
        genFaceEmbdng.genFaceEmbedding()

def trainModel():
    # ============================================= Training Params ====================================================== #

    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle",
                    help="path to serialized db of facial embeddings")
    ap.add_argument("--model", default="faceEmbeddingModels/my_model.h5",
                    help="path to output trained model")
    ap.add_argument("--le", default="faceEmbeddingModels/le.pickle",
                    help="path to output label encoder")
    ap.add_argument("--training", required=True,help="True if training required or False for inferencing")

    args = vars(ap.parse_args())

    faceRecogModel = TrainFaceRecogModel(args)
    faceRecogModel.trainKerasModelForFaceRecognition()

def makePrediction():
        faceDetector = FacePredictor()
        faceDetector.detectFace()

if __name__=="__main__":
    train=collect_images()
    if train=="True":
        getFaceEmbedding()
        trainModel()
    else:
        faceDetector=FacePredictor()
        faceDetector.detectFace()




