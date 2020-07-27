import numpy as np
import pickle
import cv2
from PIL import Image as img
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from mtcnn import MTCNN

data = np.load('trainner2.npz')
trainX = data['arr_0']
id_ = data['arr_1']
trainX = trainX[:,0,:]

detector = MTCNN()

labels = {'person_name':1}
with open('labels.pickle', 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

#cap = cv2.VideoCapture('rtsp://admin:dadDy$12A@192.168.1.11')
cap = cv2.VideoCapture(r'D:\randomshit\2020-02-21\FR3.mp4')
#cap = cv2.VideoCapture(0)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
out_encoder = LabelEncoder()
out_encoder.fit(id_)
id_ = out_encoder.transform(id_)

model = SVC(kernel='linear', probability=True)
model.fit(trainX, id_)

while True:
	ret, frame = cap.read()
	res = detector.detect_faces(frame)
	font = cv2.FONT_HERSHEY_SIMPLEX
	ndd = frame[:,:,0]
	if res:
		fbx = res[0]['box']
		x,y,w,h = fbx[0], fbx[1], fbx[2], fbx[3]
		ax = abs(x)
		ay = abs(y)
		roi = ndd[ay:ay+h, ax:ax+h]
		pic = img.fromarray(roi)
		pic = pic.resize((128,128))
		picarr = np.asarray(pic)
		yhat_class = model.predict(picarr)
		yhat_prob = model.predict_proba(picarr)

		class_index = yhat_class[0]
		conf = yhat_prob[0, class_index] * 100

		if conf >= 70:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
			cv2.putText(frame, labels[class_index], (x-5, y-5), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
			print(str(conf))
		else:
			un = 'Unknown'
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
			cv2.putText(frame, un, (x-5, y-5), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
	else:
		pass

	frame = rescale_frame(frame, percent = 50)
	cv2.imshow('MTCNN Facial Recognition', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
