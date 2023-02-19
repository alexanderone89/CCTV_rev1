#! /usr/bin/python3
# coding=utf-8


import mainwindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QMainWindow, QAction, qApp
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from PyQt5.QtGui import QPixmap
import glob

import queue
import threading

import datetime
import time
# from imutils.video import VideoStream
# from imutils.video import FPS
import imutils
import os
# import time


url1 = 'rtsp://192.168.1.23:554/user=admin_password=tlJwpbo6_channel=1_stream=1.sdp?real_stream'
# url1 = 'rtsp://192.168.1.23:554/stream1'
url2 = 'rtsp://192.168.1.11:554/user=admin_password=tlJwpbo6_channel=1_stream=1.sdp?real_stream'
url3 = 'rtsp://192.168.1.12:554/user=admin_password=tlJwpbo6_channel=1_stream=1.sdp?real_stream'
url4 = 'rtsp://192.168.1.199:554/user=admin_password=tlJwpbo6_channel=1_stream=1.sdp?real_stream'




CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]


# MY_CLASSES = [6, 7, 8, 12, 15]
# COLORS = np.random.uniform(0, 210, size=(len(CLASSES), 3))

path_for_screenshots = os.path.dirname(os.path.realpath(__file__)) + '/screenShots-'
print(path_for_screenshots)

que = queue.Queue()

class Detected(QThread):


    change_pixmap_signal = pyqtSignal(np.ndarray)


    def __init__(self):
        super().__init__()
        self._run_flag = True

# парапрпа

    def run(self):
        scale_ratio = 1
        

        # net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")


        # hog = cv2.HOGDescriptor()
        # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


        # while self._run_flag == True:
        #     if not que.empty():
        #         # print(f"[+] длина очереди = {que.qsize()}")
        #         print(type(que.get()))
        #         # image = que.get()



        #         wd = int(image.shape[1] * scale_ratio)
        #         hg = int(image.shape[0] * scale_ratio)
        #         frm = cv2.resize(image, (wd, hg))
        #         gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        #         (found, _w) = hog.detectMultiScale(gray, winStride=(16,16), padding=(4,4), scale=1.05)
        #         if len(found) > 0:
        #             print(f"[+] detected  {len(found)} objects")
        #             for (x,y,w,h) in found:
        #                 cv2.rectangle(image, (x,y), (x + w, y + h), (0, 0, 255), 4)


                        # nowtime = datetime.datetime.now()
                        # filenamee = f"img_{nowtime.year}_{nowtime.month}_{nowtime.day}_{nowtime.hour}_{nowtime.minute}_{nowtime.second}_{nowtime.microsecond}.png"
                        # cv2.imwrite('/home/user1/www1/dataset/'+filenamee, image[(x, y), (x + w, y + h)])

                
                    # frame = cv2.resize(image, (1080 // 2, 600 // 2))
                    # self.change_pixmap_signal.emit(frame)

                    # time.sleep(0.01)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    clear_pixmap_label   = pyqtSignal(bool)



    def __init__(self, url, num):
        super().__init__()
        self._run_flag = True
        self.url = url
        self.num = num
        self.screenShot = False

        self.recognition_people = True
        self.recognition_cars   = False



    def run(self):

        # files_pickle = glob.glob('*.pickle')

        # net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

        # vs = VideoStream(self.url).start()
        # time.sleep(2.0)
        # fps = FPS().start()
        # while self._run_flag:
        #     frame = vs.read()
        #     # frame = imutils.resize(frame, width=400)
        #     (h, w) = frame.shape[:2]
        #     blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        #     net.setInput(blob)
        #     detections = net.forward()
        #     for i in np.arange(0, detections.shape[2]):
        #         confidence = detections[0, 0, i, 2]
        #         # print(f"[+]   confidence {confidence}")
        #         if confidence >float(0.75):
        #             # print(f"[+]   -------------------")
        #             idx = int(detections[0, 0, i, 1])
                    
        #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #             (startX, startY, endX, endY) = box.astype("int")
        #             label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        #             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255) , 2) # COLORS[idx], 2)
        #             y = startY - 15 if startY - 15 > 15 else startY + 15
        #             # print(f"[+]           {y}")
        #             cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # COLORS[idx], 2)
        #     fps.update()
        #     self.change_pixmap_signal.emit(frame)
            
        # fps.stop()
        # vs.stop()            

        # hog = cv2.HOGDescriptor()
        # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        cap = cv2.VideoCapture(self.url)
        # rt1, frm1 = cap.read()
        # rt2, frm2 = cap.read()


        # prvs = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)
        # hsv = np.zeros_like(frm1)
        # hsv[..., 1] = 255

        # kernel = np.ones((3, 3), 'uint8')
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))


        # feature_params = dict(maxCorners = 100, 
        #                 qualityLevel = 0.3,
        #                 minDistance = 7,
        #                 blockSize = 7)
        # lk_params = dict(winSize = (15,  15),
        #                         maxLevel = 2,
        #                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # color = np.random.randint(0, 255, (100, 3))
        # ret, old_frame = cap.read()
        # old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # mask = np.zeros_like(old_frame)

        # print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # cap.set(cv2.CAP_PROP_FPS, 25)
        # print(int(cap.get(cv2.CAP_PROP_FPS)))

        scale_ratio = 1
        while self._run_flag:
        # while True:            
            try:
                ret, cv_img = cap.read()
                if ret:
                    if self.screenShot:
                        directory = path_for_screenshots+self.num


                        try:
                            os.makedirs(directory,mode=0o777, exist_ok = True)
                            

                            nowtime = datetime.datetime.now()
                            filenamee = f"img_{nowtime.year}_{nowtime.month}_{nowtime.day}_{nowtime.hour}_{nowtime.minute}_{nowtime.second}_{nowtime.microsecond}.png"
                            # print(directory+filenamee)


                            path = os.path.join(directory, filenamee)
                            # print(path)
                            cv2.imwrite(path, cv_img)
                            # cv2.imwrite('/home/user1/www1/dataset/'+filenamee, cv_img) 
                            print("Screenshot '%s' created successfully" %path)                      
                        except OSError as error:
                            print(f"Screenshot  can not be created {error}")
                            self.screenShot = False
                        finally:
                            self.screenShot = False

                    if int(self.num) == 1:
                        num_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if num_frame % 10 == 0:
                            pass
                            # que.put(cv_img)
                            # print(f"[+] put to Queue")


                            # wd = int(cv_img.shape[1] * scale_ratio)
                            # hg = int(cv_img.shape[0] * scale_ratio)
                            # frm = cv2.resize(cv_img, (wd, hg))
                            # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                            # (found, _w) = hog.detectMultiScale(gray, winStride=(16,16), padding=(4,4), scale=1.05)
                            # for (x,y,w,h) in found:
                            #     cv2.rectangle(cv_img, (x,y), (x + w, y + h), (0, 0, 255), 4)
                            # print(f"frame= {num_frame}       found object= {len(found)}")


                    # if (not int(self.num) == 4) and (not int(self.num) == 2):
                    # if  int(self.num) == 1:
                    #     # # + 1 обводим контур движ оббектов
                    #     difference = cv2.absdiff(frm1, frm2)
                    #     gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
                    #     blur = cv2.GaussianBlur(gray, (5,5), 0)
                        # _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                        # dilate = cv2.dilate(threshold, kernel, iterations=5)
                        # contours , _ =cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # for contour in contours: 
                        #         (x,y,w,h) = cv2.boundingRect(contour)
                        #         if cv2.contourArea(contour) > 500:
                        #             cv2.rectangle(frm1, (x, y), (x + w, y + h), (255, 255, 0), 1)

                        # edged = cv2.Canny(blur, 10, 250)
                        
                        # closed_img = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
                        # contours =cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # cnts = imutils.grab_contours(contours)
                      
                        # for cnt in cnts:
                        #     peri = cv2.arcLength(cnt, True)
                        #     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                        #     if len(approx) > 12  :
                        #         ss = cv2.contourArea(cnt)
                        #         if ss > 400:
                        #             (x,y,w,h) = cv2.boundingRect(cnt)
                        #             cv2.rectangle(frm1, (x, y), (x + w, y + h), ( 0, 0, 255), 5)
                        #             # cv2.drawContours(frm1, [approx], -1,  ( 0, 0, 255), 5)
                        #             print(f"[+] камера-{self.num}   количество вершин = {len(approx)}     прощадь = {ss}")



# nowtime = datetime.datetime.now()
# filenamee = f"img_{nowtime.year}_{nowtime.month}_{nowtime.day}_{nowtime.hour}_{nowtime.minute}_{nowtime.second}_{nowtime.microsecond}.png"
# cv2.imwrite('/home/user1/www1/dataset/'+filenamee, frm1[(x, y), (x + w, y + h)])



# +
                    # wd = int(cv_img.shape[1] * scale_ratio)
                    # hg = int(cv_img.shape[0] * scale_ratio)
                    # frm = cv2.resize(cv_img, (wd, hg))
                    # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    # (found, _w) = hog.detectMultiScale(gray, winStride=(16,16), padding=(4,4), scale=1.05)
                    # for (x,y,w,h) in found:
                    #     cv2.rectangle(cv_img, (x,y), (x + w, y + h), (0, 0, 255), 4)
                    # print(len(found))


                    # frame_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    # good_new = p1[st==1]
                    # good_old = p0[st==1]
                    # for i,(new, old) in enumerate(zip(good_new, good_old)):
                    #     a,b = new.ravel()
                    #     c,d = old.ravel()
                    #     # print(f"[+]   {color[i].tolist()}")
                    #     mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), color[i].tolist(), 2)
                    #     # print("OK")                        
                    #     cv_img = cv2.circle(cv_img, (int(a),int(b)), 5, color[i].tolist(), -1)
                    # img = cv2.add(cv_img, mask)

                    

                    frame = cv2.resize(cv_img, (1080 // 2, 600 // 2))
                    self.change_pixmap_signal.emit(frame)


                    # old_gray = frame_gray.copy()
                    # p0 = good_new.reshape(-1, 1, 2)

                    # prvs = nextt

                    # frm1 = frm2
                    # frm2 = cv_img
                else:
                    print("[-]            !!!!!!")
                    cap = cv2.VideoCapture(self.url)

            except:
                self.clear_pixmap_label.emit(False)
                print("[--]            !!!!!!")
                # pass

        # print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()



    def compare_faces(self, img1_path, img2_path):
        img1 = face_recognition.load_image_file(img1_path)
        img1_encodings = face_recognition.face_encodings(img1)[0]

        img2 = face_recognition.load_image_file(img2_path)
        img2_encodings = face_recognition.face_encodings(img2)[0]

        result = face_recognition.compare_faces([img1_encodings], img2_encodings)
        # print(result)
        return result[0]


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.recognition_people = False


        # self.det.stop()



        self.wait()
        

    def recognition_people(self, status):
        self.recognition_people  = status

    def recognition_cars(self, status):        
        self.recognition_cars   = status



class AppForm(QtWidgets.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        exitAction = QAction("Выход", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.triggered.connect(qApp.quit)
        self.mainToolBar = self.addToolBar("Exit")
        self.mainToolBar.addAction(exitAction)

        self.thread1 = VideoThread(url1, "1")
        self.thread1.change_pixmap_signal.connect(self.update_label1)
        self.thread1.clear_pixmap_label.connect(self.clear_label1)
        self.thread1.start()
        
        self.thread2 = VideoThread(url2, "2")
        self.thread2.change_pixmap_signal.connect(self.update_label2)
        self.thread2.start()

        self.thread3 = VideoThread(url3, "3")
        self.thread3.change_pixmap_signal.connect(self.update_label3)
        self.thread3.start()

        self.thread4 = VideoThread(url4, "4")
        self.thread4.change_pixmap_signal.connect(self.update_label4)
        self.thread4.start()  


        self.detect1 = Detected()
        self.detect1.change_pixmap_signal.connect(self.update_label1)
        self.detect1.start()


    @pyqtSlot(np.ndarray, )
    def update_label1(self, cv_img):


        # summ = sum(sum(sum(cv_img)))
        # print(summ)
        # # if cv_img.all:
        # if summ != 0:
        qt_img = self.convert_cv_qt(cv_img)
        self.label_1.setPixmap(qt_img)
        # else:
        #     print(cv_img)
        #     self.label_1.clear()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label_1.size(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    @pyqtSlot(np.ndarray, )
    def update_label2(self, cv_img):
        qt_img = self.convert_cv_qt2(cv_img)
        self.label_2.setPixmap(qt_img) 

    def convert_cv_qt2(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label_2.size(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(np.ndarray, )
    def update_label3(self, cv_img):
        qt_img = self.convert_cv_qt3(cv_img)
        self.label_3.setPixmap(qt_img) 

    def convert_cv_qt3(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label_3.size(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    @pyqtSlot(np.ndarray, )
    def update_label4(self, cv_img):
        qt_img = self.convert_cv_qt4(cv_img)
        self.label_4.setPixmap(qt_img) 

    def convert_cv_qt4(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label_4.size(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    @pyqtSlot(bool)
    def clear_label1(labclear):
        if not labclear:
            self.label_1.clear()


    def closeEvent(self, event):
        self.thread1.stop()
        self.thread2.stop()
        self.thread3.stop()
        self.thread4.stop()
        self.detect1.stop()
        event.accept()


    def keyPressEvent(self, event):
        modifiers = event.modifiers()

        if ((modifiers & QtCore.Qt.ControlModifier) and event.key() == QtCore.Qt.Key_1):
            self.thread1.screenShot = True
            print(f'Нажата  комбинация клавиш Ctrl+1')

        if ((modifiers & QtCore.Qt.ControlModifier) and event.key() == QtCore.Qt.Key_2):
            self.thread2.screenShot = True
            print(f'Нажата  комбинация клавиш Ctrl+2')

        if ((modifiers & QtCore.Qt.ControlModifier) and event.key() == QtCore.Qt.Key_3):
            self.thread3.screenShot = True
            print(f'Нажата  комбинация клавиш Ctrl+3')

        if ((modifiers & QtCore.Qt.ControlModifier) and event.key() == QtCore.Qt.Key_4):
            self.thread4.screenShot = True
            print(f'Нажата  комбинация клавиш Ctrl+4')
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = AppForm()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()            