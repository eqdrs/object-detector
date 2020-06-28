from __future__ import print_function
from pipeline.utils import dataset
from pipeline.utils import Conf
from sklearn.svm import SVC
import numpy as np
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required = True,
    help = "Diretório do arquivo de configuração")
ap.add_argument("-n", "--hard_negatives", type = int, default = -1,
    help = "Flag indicando se o Hard Negative Mining deverá ser usado")
args = vars(ap.parse_args())

print("-> Carregando dataset...")
conf = Conf(args["conf"])
(data, labels) = dataset.load_dataset(conf["features_path"], "features")

if args["hard_negatives"] > 0:
    print("-> Carregando imagens negativas...")
    (hardData, hardLabels) = dataset.load_dataset(conf["features_path"], "hard_negatives")
    data = np.vstack([data, hardData])
    labels = np.hstack([labels, hardLabels])

print("-> Treinando classificador...")
model = SVC(kernel = "linear", C = conf["C"], probability = True, random_state = 42)
model.fit(data, labels)

print("-> Gravando classificador em arquivo...")
f = open(conf["classifier_path"], "wb")
f.write(pickle.dumps(model))
f.close()
