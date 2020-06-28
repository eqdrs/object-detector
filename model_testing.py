from pipeline.object_detection import ObjectDetector
from pipeline.descriptors import HOG
from pipeline.utils import Conf
import imutils
import argparse
import pickle
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "Diretório do arquivo de configuração")
ap.add_argument("-i", "--image", required=True, help = "Diretório da imagem a ser classificada")
args = vars(ap.parse_args())

conf = Conf(args["conf"])

model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations = conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]),
	cellsPerBlock = tuple(conf["cells_per_block"]), normalize = conf["normalize"], block_norm = "L1")
od = ObjectDetector(model, hog)

image = cv2.imread(args["image"])
image = imutils.resize(image, width = min(260, image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(boxes, probs) = od.detect(gray, conf["sliding_window_dim"], winStep = conf["window_step"],
	pyramidScale = conf["pyramid_scale"], minProb = conf["min_probability"])

for (startX, startY, endX, endY) in boxes:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

cv2.imshow("Imagem =", image)
cv2.waitKey(0)
