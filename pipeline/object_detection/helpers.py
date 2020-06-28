import imutils
import cv2

def crop_ct101_bb(image, bb, padding = 10, dstSize = (32, 32)):
    # pega os valores da bounding box, extrai o ROI da imagem, levando em consideração
    # o offset fornecido
    (y, h, x, w) = bb
    (x, y) = (max(x - padding, 0), max(y - padding, 0))
    roi = image[y : h + padding, x : w + padding]

    # redimensiona o ROI para as dimensões de destino desejadas
    roi = cv2.resize(roi, dstSize, interpolation = cv2.INTER_AREA)

    return roi

def pyramid(image, scale = 1.5, minSize = (30, 30)):
    # fornece a imagem original
    yield image

    # loop através da pirâmide
    while True:
        # calcula as novas dimensões da imagem e então redimensiona a mesma
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width = w)

        # se a imagem redimensionada não atender ao tamanho mínimo
        # fornecido, então interrompe a construção da pirâmide
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # fornece a próxima imagem para a pirâmide
        yield image

def sliding_window(image, stepSize, windowSize):
    # desliza uma janela através da imagem
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # fornece a janela atual
            yield (x, y, image[y : y + windowSize[1], x : x + windowSize[0]])
            
