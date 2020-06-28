from __future__ import print_function
from pipeline.object_detection import ObjectDetector
from pipeline.descriptors import HOG
from pipeline.utils import dataset
from pipeline.utils import Conf
from imutils import paths
import numpy as np
import progressbar
import argparse
import pickle
import random
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True, help = "Diretório do arquivo de configuração")
args = vars(ap.parse_args())

conf = Conf(args["conf"])
data = []

model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = HOG(orientations = conf["orientations"], pixelsPerCell = tuple(conf["pixels_per_cell"]),
    cellsPerBlock = tuple(conf["cells_per_block"]), normalize = ["normalize"], block_norm = "L1")
od = ObjectDetector(model, hog)

dstPaths = list(paths.list_images(conf["image_distractions"]))
dstPaths = random.sample(dstPaths, conf["hn_num_distraction_images"])

widgets = ["Mining: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval = len(dstPaths), widgets = widgets).start()

for (i, imagePath) in enumerate(dstPaths):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (boxes, probs) = od.detect(gray, conf["sliding_window_dim"], winStep = conf["hn_window_step"],
        pyramidScale = conf["hn_pyramid_scale"], minProb = conf["hn_min_probability"])

    for (prob, (startX, startY, endX, endY)) in zip(probs, boxes):
        roi = cv2.resize(gray[startY: endY, startX: endX], tuple(conf["sliding_window_dim"]),
            interpolation = cv2.INTER_AREA)
        features = hog.describe(roi)
        data.append(np.hstack([[prob], features]))

    pbar.update(i)

pbar.finish()
print("-> Ordenando por probabilidade...")
data = np.array(data)
data = data[data[:, 0].argsort()[::-1]]

print("-> Gravando dados do hard negative mining em arquivo...")
dataset.dump_dataset(data[:, 1:], [-1] * len(data), conf["features_path"], "hard_negatives",
    writeMethod = "a")

