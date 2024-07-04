# Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring
Developed a driver drowsiness detection system utilizing Convolutional Neural Networks (CNN) for superior accuracy and robustness compared to traditional computer vision techniques.

**INTRODUCTION**
•Driver drowsiness is a critical factor contributing to road accidents worldwide. Fatigue and drowsiness impair a driver's cognitive abilities, reaction time, and decision-making skills, leading to an increased risk of collisions and loss of lives.To address this pressing issue and enhance road safety, this project introduces a real-time driver drowsiness detection system using Convolutional Neural Networks (CNNs) and dlib for eye detection.The system captures live video from the driver's webcam, locates the eyes using dlib's facial landmark detection, and feeds eye regions to a pre-trained CNN model.The CNN effectively classifies eye states (open or closed) to determine the driver's alertness level. Challenges include efficient eye detection, low latency processing, and robustness in varying conditions.We will present the system's development steps, including data collection, CNN training, eye detection integration, and real-time evaluation.

**Problem Statement:**
•The problem addressed in this project is driver drowsiness, which poses a significant threat to road safety. Driver drowsiness is a critical issue affecting road safety, leading to an increased risk of accidents and fatalities worldwide.Fatigue and drowsiness impair a driver's cognitive abilities, reaction time, and decision-making skills, making them prone to lapses in attention and reduced alertness.Therefore, there is a pressing need to develop an innovative and robust driver drowsiness detection system that can accurately and efficiently monitor a driver's alertness level in real-time.The proposed system leverage advanced technologies such as computer vision i.e. Web cam, machine learning algorithm provide timely alerts and prevent potential accidents caused by drowsy driving.

**Objective of this project:**
•The main objective of this project to build model using deep learning techniques which helps to identify whether ther person is drowsy or not.The application helps in detecting eye of the person and alarm if the eye is closed for 10 seconds.The goal of the project is to enhance road safety by providing timely alerts to prevent accidents.
In this project,
we are going to make use of web camera for the detection purpose. In the proposed system we are going to apply cnn algorithm fr tracking the eye images and calculating the frames for 10 seconds, if the eye is closed for 10 seconds the system will going to play an alarm which indicates the person is in sleepy mode.

**Advantages of this project**

•By adapting convolution neural network is that its ability to integrate distinct categories of parameters was much better.

•Early detection of drowsiness reduces the likelihood of accidents, ultimately leading to fewer fatalities and injuries on the road.

**Working of the model:**
Driver exhaustion and drowsiness are significant contributors to various automobile accidents. In the field of accident prevention systems, designing and maintaining technology that can effectively identify or avoid drowsiness at the wheel and warn the driver before a collision is a major challenge. We use OpenCV to take images from a webcam and these images given to a deep learning algorithm that can tell whether someone's eyes are closed or opened. In this case, we are looking for the persons face and eyes.

**Step1:** **Image is taken as input from camera.**
        We'll use a camera to capture photographs as input. But, in order to gain access to the webcam, we created an endless loop that captures each frame. We employ the cv2 method given by OpenCV. VideoCapture(0) (cap) is used to access the camera and capture the object. With cap.read(), each frame is read, and then image is saved in a variable.
				
**Step2:** **Create a ROI by detecting a face in the picture.**
        To segment the face in the captured image, we first converted it to gray scale because, the OpenCV object detection algorithm only accepts grayscale images as input. To detect the objects, we don't need colour detail. We use the Haar cascade classifier to detect the face. The classifier face= cv2.Cas is set with this section. for (x,y,w,h)interfaces,
        we use cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100), 1
				
**Step 3:** **Use the ROI to find the eyes and feed them to the classifier.**
        The technique for detecting eyes is the same as for detecting ears. Cascade classifier is used in left and right eyes.Then, use left_eye=leye.detectMultiScale(gray) to detect the eyes. We extracted only the details of eyes from the captured image. This can be done by first removingthe eye's boundary box and then using this code to remove the eye image from the picture.
l_eye = frame[y : y+h, x : x+w].This information is given to CNN, which decides whether the eyes are closed or not. The right eye also detected in the above manner.
![hhh](https://github.com/Vignesh3122/Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring/assets/146365068/46bb7930-0a43-4b8e-ac0d-89438d9618da)
Figure1: Detection of Eyes using OpenCV

**Step 4**: **The classifier will determine whether or not the eyes are open:**
The eye status is predicted using a CNN classifier to feed the image into the model, since the model requires the proper measurements to begin with. We begin by converting the colour picture to grayscale.
r_eye=cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY).
Then, since the model is trained on images with a resolution of 24*24 pixels, We resize the image to 24*24pixels.
cv2.resize (r_eye, (24,24)).
For better convergence, the date is normalized.
r_eye = r_eye/255
The model is loaded using
model=load_model(„models/cnnCat2.h5‟)
Now, each eye is predicted with the proposed model.
lpred=model.predict_classes(l_eye)
If lpred[0] = 1,
it means that eyes are open,
if lpred[0] = 0 then,
it means that eyes are closed.

**Step 5:** **Score Calculation**
The score is essentially a number that we'll use to figure out how long the individual has been closed-eyed. As a consequence, if both eyes are closed, we will begin to raise the score, but if both eyes are open, we will decrease the score. We're using the cv2.putText() function to draw the result on the screen, which displays the status of the driver or a person.
cv2.putTxt(frame,“Open”,(10,height20), font,1,(255,255,255),1,cv2.LINE_AA )
A criterion is established, for example, if the score exceeds 15, it indicates that the person's eyes have been closed for an extended amount of time. Then the alarm turned on.

**Step 6: To run the file**
You can either open the command prompt or navigate to the directory containing our key file

**Architecture:**
![wwwwwwwwwwwww](https://github.com/Vignesh3122/Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring/assets/146365068/84608fe0-c978-4d26-b8d6-efe2214b610b)

**How to run this project:**

1)Download the zip file of this project and save it in a directory and also download the dataset from kaggle which will contain 2 folders (Closed and Open).

2)Download python updated version from their official website.

3)Download the Dlib, OpenCv and also anaconda naviagtor

4)Set Up the anaconda environment.

5)In Jupyter command Prompt 
	wirte "conda Drowsiness.py"
 
6)the project will run.

7)in order to stop the execution press "esc" button.

**Screenshots:**


![image](https://github.com/Vignesh3122/Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring/assets/146365068/d87ae0ee-d8a5-48ad-b411-d6f45dec77b9)

Opened Eye Dataset

![image](https://github.com/Vignesh3122/Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring/assets/146365068/f936e1bd-ecf1-4c2b-861c-d38cb9740306)

Closed Eye Dataset

![image](https://github.com/Vignesh3122/Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring/assets/146365068/4a806e76-a2fb-4e2b-8974-38b0126e72d3)

Model Prediction Screenshot

a)Closed Eye predicted and alert
![image](https://github.com/Vignesh3122/Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring/assets/146365068/4cbf3af7-da6b-4e9e-9f37-6bac07518b1d)

![image](https://github.com/Vignesh3122/Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring/assets/146365068/85ab036a-978d-4578-8e2f-d29576a3619b)

b)Open Eye Predicted 

![image](https://github.com/Vignesh3122/Image-Processing-Method-For-Drowsy-Driver-Alertness-Monitoring/assets/146365068/7062b120-4645-48ce-8c14-bef721a0fdbb)

