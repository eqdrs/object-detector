from pipeline.object_detection import non_max_suppression
from pipeline.object_detection import bb_intersection_over_union
from pipeline.object_detection import ObjectDetector
from pipeline.descriptors import HOG
from pipeline.utils import Conf
import numpy as np
import imutils
import argparse
import pickle
import cv2

from scipy import io
import glob
import pdb

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "Diretório do arquivo de configuração")
ap.add_argument("-i", "--image", required = True, help = "Diretório da imagem a ser classificada")
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
pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])
orig = image.copy()

for (startX, startY, endX, endY) in boxes:
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

for (startX, startY, endX, endY) in pick:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

#=====================================================================================

# # plotando o box da ground-truth
# path = args["image"].split('/')
# annotation_name = path[-1].replace('jpg', 'mat').replace('image', 'annotation')

# annotation_dir = conf["image_annotations"] + "/" + annotation_name
# (y, h, x, w) = io.loadmat(annotation_dir)["box_coord"][0]
# (y, h, x, w) = [y-15, h-19, x,  w-36]
# cv2.rectangle(image, (x, y), (w, h), (127, 0, 255), 2)

# # calculando IoU
# predicted_box = pick[0]
# iou = bb_intersection_over_union([x, y, w, h], predicted_box)

# # printando valor do IoU na imagem
# cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
#   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 0, 255), 2)

#=====================================================================================

cv2.imshow("Original (sem NMS)", orig)
cv2.imshow("Com Hard Negative Mining", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
