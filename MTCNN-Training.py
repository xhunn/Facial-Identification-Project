import os
import pickle
import numpy as np
from PIL import Image as img
import cv2
from mtcnn import MTCNN
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'Face-Records')

face_cascade = MTCNN()
#recognizer = cv2.face.LBPHFaceRecognizer_create()

label_ids = {}
current_id = 0
y_label = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()

			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]

			pil_image = img.open(path).convert('RGB') #Converts to Grayscale
			#pil_image = pil_image.resize((160,160))
			image_array = np.asarray(pil_image)
			faces = face_cascade.detect_faces(image_array)
			for n in faces:
				x,y,w,h = faces[0]['box']
				x = abs(x)
				y = abs(y)
				roi = image_array[y:y+h, x:x+w]
				y_label.append(id_)
				image = img.fromarray(roi)
				image = image.resize((160,160))
				f_array = np.asarray(image)
				print(label)
				print(file)
				fpix = f_array.astype('float32')
				mean = fpix.mean()
				strd = fpix.std()
				fpix = (fpix - mean)/strd
				samples = np.expand_dims(fpix, axis=0)
				model = load_model(r'C:\Users\Kent\AppData\Local\Programs\Python\Python37\Lib\site-packages\facenet_keras.h5')
				yhat = model.predict(samples)
				x_train.append(yhat)
				print('Training done for 1 image of: ' + str(label))

with open('labels.pickle', 'wb') as f:
	pickle.dump(label_ids, f)

y_train = np.array(y_label)
x_train = np.asarray(x_train)
np.savez_compressed('trainner2', x_train, y_train)
