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
import time

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "Diretório do arquivo de configuração")
args = vars(ap.parse_args())

start_time = time.time()

conf = Conf(args["conf"])

model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations = conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]),
  cellsPerBlock = tuple(conf["cells_per_block"]), normalize = conf["normalize"], block_norm = "L1")
od = ObjectDetector(model, hog)

iou = []

for image_path in glob.glob(conf["image_dataset"] + "2/*.jpg"):
  print(image_path)
  image = cv2.imread(image_path)
  image = imutils.resize(image, width = min(260, image.shape[1]))
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  (boxes, probs) = od.detect(gray, conf["sliding_window_dim"], winStep = conf["window_step"],
    pyramidScale = conf["pyramid_scale"], minProb = conf["min_probability"])
  pick = non_max_suppression(np.array(boxes), probs, conf["overlap_thresh"])

  path = image_path.split('/')
  annotation_name = path[-1].replace('jpg', 'mat').replace('image', 'annotation')

  annotation_dir = conf["image_annotations"] + "/" + annotation_name
  (y, h, x, w) = io.loadmat(annotation_dir)["box_coord"][0]
  (y, h, x, w) = [y-15, h-19, x,  w-36]

  try:
    predicted_box = pick[0]
    iou_result = bb_intersection_over_union([x, y, w, h], predicted_box)
    iou.append(iou_result)
  except:
    print('FALSO NEGATIVO!')
    iou.append('*****FALSO-NEGATIVO*****')

print(iou)

print("--- %s seconds ---" % (time.time() - start_time))

