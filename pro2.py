import cv2
cascade = cv2.CascadeClassifier('E:\\Project\\msme_project\\face.xml')

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('E:\\Project\\msme_project\\pic2.yml')

names = ['x','Abhranil']
cam = cv2.VideoCapture(0)
 
while True :
	a,frame = cam.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = cascade.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces :
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		try:
			id,con = recogniser.predict(gray[y:y+h,x:x+w])
			print(con)
			if con > 60 :

				name = names[id]
			else :
				name = 'not found'	
		except Exception as e:
			print(e)
			name = "error"
		cv2.putText(frame,name,(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
	cv2.imshow('ab',frame)
	if cv2.waitKey(1)==27:
		break
cam.release()
cv2.destroyAllWindows()