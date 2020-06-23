# -*- coding: utf-8 -*-
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QTableWidgetItem, QFileDialog,QWidget, QDesktopWidget
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import os

configPath = "yolov3.cfg"
weightsPath = "yolov3.weights"
classesPath = "yolov3.txt"


class App(QWidget):
    
    def __init__(self):
        super(App,self).__init__()
        
        Form, Window = uic.loadUiType("mainWin.ui")
        window = Window()
        form = Form()
        form.setupUi(window)
        window.show()
                
        self.im_counter = 0
        self.scale = 0.00392
        self.classes = None
        with open(classesPath, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))        
        self.net = cv2.dnn.readNet(weightsPath,configPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.savedImages= []
        self.seen = set()
        
        self.label = form.loadedImgLbl
        self.quitBut = form.qbtn.clicked.connect(self.quitApp)
        self.loadImButton = form.loadIm.clicked.connect(self.loadImageFunction)
        self.loadVidButton = form.loadVid.clicked.connect(self.loadVideoFunction)
        self.updateBoxes = form.updateBoxes
        self.updateBoxes.clicked.connect(self.updateBoxesFunction)        
        self.showLibButton = form.saveToLib
        self.showLibButton.clicked.connect(self.saveToLibFunction)
        self.detTable = form.detailsTable
        
        app.exec_()
    
    def updateBoxesFunction(self):
        self.image = cv2.imread(self.loadedImage)
        for i in range(1,self.rows):            
            label = (self.detTable.item(i,0).text())
            x = int(self.detTable.item(i,1).text())
            y = int(self.detTable.item(i,2).text())
            w = int(self.detTable.item(i,3).text())
            h = int(self.detTable.item(i,4).text())
            self.draw_bounding_box(self.image, label,self.classes_ids[i-1], x, y, (x+w), (y+h))
            self.label.setPixmap(QPixmap('object-detection.jpg'))
        
    def quitApp(self):
        QApplication.quit()
        
    def saveToLibFunction(self):
        
        if self.loadedImage in self.seen:
            print ("Already in library")
        else:
            self.seen.add(self.loadedImage)
            self.savedImages.append({'id':self.im_counter, 'loaded_path': self.loadedImage, 'indices':self.im_indices,
                                    'boxes':self.im_boxes,'classes_ids':self.classes_ids})
            self.im_counter += 1
            print(self.savedImages)
        
    def loadVideoFunction(self):
        self.label.setText('Loading....')  
        
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        self.loadedVideo = QFileDialog.getOpenFileName(None,'Open file',desktop)[0]
        
        self.pixmap = QPixmap(self.loadedVideo)        
        self.label.setScaledContents(True)
        self.label.setPixmap(self.pixmap)

        pixmapWidth = 660*0.71
        pixmapHeight = 660*0.71
        self.label.resize(pixmapWidth,pixmapHeight)
        self.label.move(660/2-pixmapWidth/2,660/2-pixmapHeight/2)
        
        cap = cv2.VideoCapture(self.loadedVideo)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = self.image .shape 
                self.blob = cv2.dnn.blobFromImage(self.image , self.scale, (416,416), (0,0,0), True, crop=False)
                self.net.setInput(self.blob)
                
                self.yoloPredImage()
        
                self.label.setPixmap(QPixmap('object-detection.jpg'))
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        cap.release()
        cv2.destroyAllWindows()
          
    def loadImageFunction(self):
        self.label.setText('Loading....')  
        
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        self.loadedImage = QFileDialog.getOpenFileName(None,'Open file',desktop)[0]
        
        self.pixmap = QPixmap(self.loadedImage)        
        self.label.setScaledContents(True)
        self.label.setPixmap(self.pixmap)
        
        self.image = cv2.imread(self.loadedImage)
        
        if (self.pixmap.width() > 660) or (self.pixmap.height() > 660):
            ratio = max(self.pixmap.width()/660,self.pixmap.height()/660) + 0.71
            self.label.resize(self.pixmap.width() / ratio,self.pixmap.height() / ratio )
            pixmapWidth = self.pixmap.width() / ratio
            pixmapHeight = self.pixmap.height() / ratio
        else:
            self.label.resize(self.pixmap.width(),self.pixmap.height())
            pixmapWidth = self.pixmap.width()
            pixmapHeight = self.pixmap.height()

        self.label.move(660/2-pixmapWidth/2,660/2-pixmapHeight/2)
        self.blob = cv2.dnn.blobFromImage(self.image, self.scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(self.blob)
        
        self.yoloPredImage()
        
        self.showLibButton.setEnabled(True)
        self.updateBoxes.setEnabled(True)
        self.label.setPixmap(QPixmap('object-detection.jpg'))
                    
    def yoloPredImage(self):
        outs = self.net.forward(self.get_output_layers())
        Width = self.image.shape[1]
        Height = self.image.shape[0]        

        # initialization
        class_ids = []
        ids = []
        labels = []
        colors = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            ids.append(class_ids[i])
            labels.append(self.classes[class_ids[i]])
            self.draw_bounding_box(self.image,str(self.classes[class_ids[i]]), class_ids[i], round(x), round(y), round(x+w), round(y+h))
        labels = list(labels)
        ids = list(ids)
        
        self.classes_ids = class_ids
        self.im_indices = indices
        self.im_boxes = boxes
        self.fillTable(ids,colors,labels,boxes)        
        
    def fillTable(self,ids,colors,labels,boxes):
        headers = ["Label", "X", "Y", "Width","Height"]
        self.col = len(headers)
        self.rows = len(ids) + 1

        self.detTable.setRowCount(self.rows)
        self.detTable.setColumnCount(self.col)
        
        for i in range(0,self.col):
            self.detTable.setItem(0,i, QTableWidgetItem(headers[i]))
        
        for i in range(1,self.rows):            
            self.detTable.setItem(i,0, QTableWidgetItem(labels[i-1]))
            for j in range(1,self.col):
                self.detTable.setItem(i,j, QTableWidgetItem(str(round(boxes[i-1][j-1]))))
            self.detTable.resizeColumnsToContents()       
        
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return self.output_layers
        
    
    def draw_bounding_box(self,img, label, class_id, x, y, x_plus_w, y_plus_h):
        self.color = self.COLORS[class_id]    
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), self.color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, 2)
        cv2.imwrite("object-detection.jpg", self.image)
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
    
