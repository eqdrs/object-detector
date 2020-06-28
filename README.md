# Detector de Objetos com SVM Linear e HOG

## Requisitos

- Anaconda com Python 3.6.5
- OpenCV

## Instruções

### Extração das features com HOG

```bash
python features_extraction.py --conf configuration/config.json
```

### Treinando o modelo com SVM Linear (sem Hard Negative Mining)

```bash
python model_training.py --conf configuration/config.json
```

### Classificando uma imagem

```bash
python nms_model_testing.py --conf configuration/config.json --image datasets/caltech101/car_side/image_0001.jpg
```

### Aplicando o Hard Negative Mining

```bash
python hard_negative_mine.py --conf configuration/config.json
```

### Treinando o modelo com SVM Linear (com Hard Negative Mining)

```bash
python model_training.py --conf configuration/config.json --hard_negatives 1
```