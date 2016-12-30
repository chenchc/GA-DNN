import csv
import skimage.transform
import numpy as np

if __name__ == '__main__':
	file = open('train.csv', 'rb')
	reader = csv.reader(file)
	head = True
	data = []
	for row in reader:
		if head:
			head = False
			continue
		data.append(row[1:])

	images = [np.array(imgList, dtype=np.float64).reshape((28, 28)) / 255.0 for imgList in data]
	smallerImages = []
	for image in images:
		smallerImages.append(skimage.transform.resize(image, (7, 7)))

	output = open('mnist.csv', 'wb')
	writer = csv.writer(output)
	for image in smallerImages:
		writer.writerow(np.ndarray.flatten(image).tolist())
