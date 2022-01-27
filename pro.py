import cv2
import numpy as np
import os
from PIL import Image

recogniser = cv2.face.LBPHFaceRecognizer_create()

def imgtrain(path):
	imgpaths = [os.path.join(path,f) for f in os.listdir(path)]
	faces=[]
	ids = []
	for imgpath in imgpaths:
		faceimg=Image.open(imgpath).convert('L')
		facenp=np.array(faceimg,'uint8')
		Id = int(os.path.split(imgpath)[-1].split('.')[1])
		faces.append(facenp)
		ids.append(Id)
		cv2.imshow('training',facenp)
		cv2.waitKey(10)
	return np.array(ids),faces
		 
Ids,faces=imgtrain('E:\\Project\\msme_project\\pic')		 
recogniser.train(faces,Ids)
recogniser.write('E:\\Project\\msme_project\\pic2.yml')
cv2.destroyAllWindows()