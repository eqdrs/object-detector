import numpy as np

def non_max_suppression(boxes, probs, overlapThresh):
    # se não há bounding boxes, retorna uma lista vazia
    if len(boxes) == 0:
        return []

    # se as bounding boxes são inteiros, converte pra float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # inicializa a lista dos índices selecionados
    pick = []

    # pega as coordenadas da bounding box
    x1 = boxes[:, 0]
    y1 = boxes[: ,1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # calcula a area das bounding boxes e ordena pelas suas respectivas probabilidades
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    # se mantém no loop enquanto ainda existirem índices na lista
    while len(idxs) > 0:
        # pega o último índice da lista de índices e adiciona o valor do índice na 
        # lista de índices selecionados
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # busca as coordenadas (x, y) mais largas para o início da bounding box e 
        # as menores coordenadas (x, y) para o fim da bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # calcula comprimento e altura da bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # calcula a tava do overlap threshold
        overlap = (w * h) / area[idxs[:last]]

        # remove todos os índices da lista de índices que possuem overlap maior que o
        # overlap threshold fornecido
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # retorna somente as bounding boxes que foram selecionadas
    return boxes[pick].astype("int")
