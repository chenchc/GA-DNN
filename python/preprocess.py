import csv
import skimage.transform
import numpy as np

if __name__ == '__main__':
	file = open('train.csv', 'rb')
	reader = csv.reader(file)
	head = True
	data = []
	labels = []
	for row in reader:
		if head:
			head = False
			continue
		data.append(row[1:])
		labels.append([0.0] * 10)
		labels[-1][int(row[0])] = 1.0

	images = [np.array(imgList, dtype=np.float64).reshape((28, 28)) / 255.0 for imgList in data]
	smallerImages = []
	for image in images:
		smallerImages.append(skimage.transform.resize(image, (7, 7)))

	output = open('mnist.csv', 'wb')
	writer = csv.writer(output)
	for image in smallerImages:
		writer.writerow(np.ndarray.flatten(image).tolist())
	
	label_output = open('mnist_label.csv', 'wb')
	label_writer = csv.writer(label_output)
	for label in labels:
		label_writer.writerow(label)

