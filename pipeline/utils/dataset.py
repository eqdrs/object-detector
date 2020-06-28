import numpy as np
import h5py

def dump_dataset(data, labels, path, datasetName, writeMethod = "w"):
	# abre o banco, cria o dataset, grava os dados e labels no dataset,
	# e então fecha o banco de dados
	db = h5py.File(path, writeMethod)
	dataset = db.create_dataset(datasetName, (len(data), len(data[0]) + 1), dtype = "float")
	dataset[0 : len(data)] = np.c_[labels, data]
	db.close()

def load_dataset(path, datasetName):
	# abre o banco, pega os labels e dados, e então fecha o dataset
	db = h5py.File(path, "r")
	(labels, data) = (db[datasetName][:, 0], db[datasetName][:, 1:])
	db.close()

	# retorna a tupla com os dados e as labels
	return (data, labels)
