from . import helpers

class ObjectDetector:

    def __init__(self, model, desc):
        # armazena o classificador e o descritor HOG
        self.model = model
        self.desc = desc

    def detect(self, image, winDim, winStep = 4, pyramidScale = 1.5, minProb = 0.7):
        # inicializa a lista de bounding boxes e probabilidades associadas
        boxes = []
        probs = []

        # loop através da pirâmide de imagens
        for layer in helpers.pyramid(image, scale = pyramidScale, minSize = winDim):
            # determina a escala atual da pirâmide
            scale = image.shape[0] / float(layer.shape[0])

            # loop através das sliding windows para a camada atual da pirâmide
            for (x, y, window) in helpers.sliding_window(layer, winStep, winDim):
                # pega as dimensões da janela
                (winH, winW) = window.shape[:2]

                # garante que as dimensões da janela coincide com as dimensões da sliding window fornecida
                if winH == winDim[1] and winW == winDim[0]:
                    # extrai as HOG features da janela atual e classifica se essa janela 
                    # contém ou não o objeto de interesse
                    features = self.desc.describe(window).reshape(1, -1)
                    prob = self.model.predict_proba(features)[0][1]

                    # checa se o classificador encontrou um objeto com probabilidade suficiente
                    if prob > minProb:
                        # calcula as coordenadas (x, y) da bounding box usando a escala atual da pirâmide de imagens
                        (startX, startY) = (int(scale * x), int(scale * y))
                        endX = int(startX + (scale * winW))
                        endY = int(startY + (scale * winH))

                        # atualiza a lista de boundig boxes e probabilidades
                        boxes.append((startX, startY, endX, endY))
                        probs.append(prob)
        # retorna uma tupla de bounding boxes e probabilidades
        return (boxes, probs)
