def bb_intersection_over_union(boxA, boxB):
	# determina as coordenadas x, y do retângulo de interseção
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# ccalcula a área do retângulo de interseção
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# calcula áa rea dos retângulos da ground-truth e da predição
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# Calcula o IoU pegando a área de interseção
	# e dividindo pelas áreas da predição + ground-truth
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# retorna o valor do IoU
	return iou