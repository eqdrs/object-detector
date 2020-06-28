from skimage import feature
import skimage

class HOG:

    def __init__(self, orientations = 12, pixelsPerCell = (4, 4), cellsPerBlock = (2, 2), normalize = True, block_norm = "L1"):
        # Armazena o número de orientações, pixels por célula, células por bloco, e
        # se será aplicada normalização à imagem
        self.orientations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalize = normalize
        self.block_norm = block_norm

    def describe(self, image):
        # Calcula as features do HOG
        hist = feature.hog(image, orientations = self.orientations, pixels_per_cell = self.pixelsPerCell,
            cells_per_block = self.cellsPerBlock, transform_sqrt = self.normalize, block_norm = self.block_norm)

        hist[hist < 0] = 0

        return hist
