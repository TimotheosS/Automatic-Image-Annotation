# -*- coding: utf-8 -*-
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QLineEdit, QInputDialog, QTableWidgetItem, QFileDialog,QWidget, QDesktopWidget, QMessageBox, QButtonGroup, QCheckBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import os
import csv

configPath = "yolov3.cfg"
weightsPath = "yolov3.weights"
classesPath = "yolov3.txt"

class App(QWidget):
    
    def __init__(self):
        super(App,self).__init__()
        
        Form, Window = uic.loadUiType("mainWin.ui")
        self.window = Window()
        form = Form()
        form.setupUi(self.window)
        self.window.show()
                
        self.updated = 1
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
        self.saveCounter = 0
        self.flag = 1
        self.segmentationFlag = 0
        self.savedPos = 0
        
        self.label = form.loadedImgLbl
        self.quitBut = form.qbtn.clicked.connect(self.quitApp)
        self.loadImButton = form.loadIm.clicked.connect(self.loadImageFunction)
        self.loadVidButton = form.loadVid.clicked.connect(self.loadVideoFunction)
        self.boundingBoxes = form.boundingBoxes
        self.boundingBoxes.clicked.connect(self.boundingBoxesFunction)
        self.segmentationButton = form.segmentationButton
        self.segmentationButton.clicked.connect(self.segmentationFunction)
        self.updateBoxes = form.updateBoxes
        self.updateBoxes.clicked.connect(self.updateBoxesFunction)  
        self.originalImage = form.originalImage
        self.originalImage.clicked.connect(self.originalImageFunction)
        self.showLibButton = form.saveToLib
        self.showLibButton.clicked.connect(self.saveToLibFunction)
        self.loadSavedButton = form.loadBtn
        self.loadSavedButton.clicked.connect(self.loadSavedFunction)
        
        self.addLabel = form.addLabel
        self.addLabel.clicked.connect(self.addLabelFunction)
        self.moveLabel = form.moveLabel
        self.moveLabel.clicked.connect(self.moveLabelFunction)
        self.adjustSizeButton = form.adjustSize
        self.adjustSizeButton.clicked.connect(self.adjustSizeFunction)
        self.deleteLabel = form.deleteLabel
        self.deleteLabel.clicked.connect(self.deleteLabelFunction)
        
        self.savedBoxes = QButtonGroup()
        self.savedBoxes.setExclusive(True)
        
        self.detTable = form.detailsTable
        self.video = 0
        
        with open('bounding_boxes.csv', mode='w') as bounding_file:
            csv.writer(bounding_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
        app.exec_()
        
    # Used to load the original image selected by the user
    def originalImageFunction(self):
        self.emptyTable()  
        self.flag = 1
        self.segmentationFlag = 0
        self.updateImage()
        
    # Used to update the shown image in the GUI
    def updateImage(self):          
        size = max(self.pixmap.height(),self.pixmap.width())
        if (self.flag == 1) and (self.segmentationFlag == 0):
            if not(self.video):
                self.image = cv2.imread(self.loadedImage)
            cv2.imwrite("object-detection.jpg", self.image)
            self.label.setPixmap(QPixmap('object-detection.jpg'))
            self.emptyTable()
        elif (self.flag == 1) and (self.segmentationFlag == 1):
            self.label.setPixmap(QPixmap('segmented-image.jpg'))
            self.emptyTable()
            self.image = self.segmImage
        elif (self.flag == 0) and (self.segmentationFlag == 0):
            if not(self.video):
                self.image = cv2.imread(self.loadedImage)
            for i in self.im_indices:
                i = i[0]
                box = self.im_boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                self.draw_bounding_box(self.image,str(self.classes[self.classes_ids[i]]), self.classes_ids[i], round(x), round(y), round(x+w), round(y+h),size/1000,2)
            cv2.imwrite("object-detection.jpg", self.image)
            self.label.setPixmap(QPixmap('object-detection.jpg'))
            self.fillTable()
        elif (self.flag == 0) and (self.segmentationFlag == 1):
            imageTmp = self.segmImage
            if self.updated:
                for i in self.im_indices:
                    i = i[0]
                    box = self.im_boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    self.draw_bounding_box(imageTmp,str(self.classes[self.classes_ids[i]]), self.classes_ids[i], round(x), round(y), round(x+w), round(y+h),size/1000,2)
            else:
                self.updateBoxesFunction()
            cv2.imwrite("object-detection.jpg", imageTmp)
            self.label.setPixmap(QPixmap('object-detection.jpg'))
            self.fillTable()
           
    # Used to add a new label as a bounding box to the image
    def addLabelFunction(self):
        self.addClicked = 1      
        self.leftClicked = 0  
        alert = QMessageBox()        
        alert.setWindowTitle("Add a new Box")
        alert.setStandardButtons(QMessageBox.Ok)      
        alert.setIcon(QMessageBox.Information)
        alert.setText('Use Left Click to set the top left corner and then, right click to set the bottom right corner.')
        alert.exec()        
        self.label.mousePressEvent = self.getPos      

    # Used to get the position that the left/right mouse click was clicked on the image and add the new label
    def getPos(self, event):
        if (event.button() == Qt.LeftButton) and (self.addClicked) and not(self.leftClicked):
            self.leftClicked = 1
            self.x_begin = round(event.pos().x() * self.ratio)
            self.y_begin = round(event.pos().y() * self.ratio)
        elif (event.button() == Qt.RightButton) and (self.addClicked) and (self.leftClicked):
            x_end = round(event.pos().x() * self.ratio)
            y_end = round(event.pos().y() * self.ratio)
            self.addClicked = 0    
            new_label = self.inputDialogLabel()
            
            if new_label in self.classes:                
                index = self.classes.index(new_label) 
                self.draw_bounding_box(self.image,str(self.classes[index]), index, round(self.x_begin), round(self.y_begin), round(x_end), round(y_end),0.8,2)                        
            else:
                index = -1
                self.draw_bounding_box(self.image,str(new_label), index, round(self.x_begin), round(self.y_begin), round(x_end), round(y_end),0.8,2)                        
            
            index = self.classes.index(new_label)            
            self.classes_ids.append(index)
            self.im_boxes.append([self.x_begin,self.y_begin, x_end-self.x_begin,y_end-self.y_begin])
            if (len(self.im_indices) == 0):
                self.im_indices = [[0]]
            else:
                self.im_indices = np.vstack((self.im_indices,len(self.im_indices)))
            self.updateImage()
            self.fillTable()                        
            
            if not(self.segmFlag):
                    self.image = cv2.imread(self.loadedImage)
                    mask = np.zeros(self.image.shape[:2],np.uint8)        
                    bgdModel = np.zeros((1,65),np.float64)
                    fgdModel = np.zeros((1,65),np.float64)
                    box = self.im_boxes[self.selectedCheckbox]
                    rect = (int(box[0]),int(box[1]),int(box[2]),int(box[3])) 
                    cv2.grabCut(self.image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
                    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                    self.segmImage = self.segmImage +  self.image*mask2[:,:,np.newaxis]
                    for y in range(0, self.image.shape[0]):
                        for x in range(0, self.image.shape[1]):
                            if (self.segmImage[y][x][0] == 0) and (self.segmImage[y][x][1] == 0) and (self.segmImage[y][x][2] == 0):
                                self.segmImage[y][x][0] = 0
                                self.segmImage[y][x][1] = 0
                                self.segmImage[y][x][2] = self.gray[y][x]
                            elif (self.segmImage[y][x][0] == 255) and (self.segmImage[y][x][1] == 255) and (self.segmImage[y][x][2] == 255):
                                self.segmImage[y][x][0] = 0
                                self.segmImage[y][x][1] = 0
                                self.segmImage[y][x][2] = 255
                    self.image = self.segmImage
                    cv2.imwrite("segmented-image.jpg", self.segmImage)                                                
            
    # Used to warn the user to add a label to the new bounding box
    def inputDialogLabel(self):
        text, ok = QInputDialog.getText(self, 'New Label Input Dialog', 'Enter a label of object:')
        le = QLineEdit()
        if ok:
            le.setText(str(text))
        return(text)
        
    # Used to move an existing bounding box of the image by pressing the left mouse click
    def moveLabelFunction(self):
        alert = QMessageBox()        
        alert.setWindowTitle("Move a Box")
        alert.setStandardButtons(QMessageBox.Ok)
        if self.selectedCheckbox == -1:
            alert.setIcon(QMessageBox.Warning)
            alert.setText('You have not selected a box to move. Please select and try again. Do not forget to press "Update" before trying to move !')
            alert.exec()
        else:
            alert.setIcon(QMessageBox.Information)
            alert.setText('Use Left Click to set the top left corner.')
            alert.exec()
            self.label.mousePressEvent = self.movePos 
            
    # Used to move the selected box to a new position in the image
    def movePos(self,event):
        if (event.button() == Qt.LeftButton):
            x_pressed = round(event.pos().x() * self.ratio)
            y_pressed = round(event.pos().y() * self.ratio)
            self.im_boxes[self.selectedCheckbox][0] = x_pressed
            self.im_boxes[self.selectedCheckbox][1] = y_pressed
        self.fillTable()
        self.updateBoxesFunction()

    # Used to adjust the size of an existing bounding box of the image
    def adjustSizeFunction(self):
        alert = QMessageBox()        
        alert.setWindowTitle("Adjust the size of a Box")
        alert.setStandardButtons(QMessageBox.Ok)
        if self.selectedCheckbox == -1:
            alert.setIcon(QMessageBox.Warning)
            alert.setText('You have not selected a box to adjust its size. Please select and try again. Do not forget to press "Update" before trying to adjust the size of a box !')
            alert.exec()
        else:
            alert.setIcon(QMessageBox.Information)
            alert.setText('Use Left Click to set the top left corner, or right click to set the bottom right corner.')
            alert.exec()
            self.label.mousePressEvent = self.adjustSize
            
    # Adjust the size of a bounding box based on which click was pressed
    def adjustSize(self,event):
        if (event.button() == Qt.LeftButton):
            x_pressed = round(event.pos().x() * self.ratio)
            y_pressed = round(event.pos().y() * self.ratio)
            self.im_boxes[self.selectedCheckbox][2] = -(x_pressed - self.im_boxes[self.selectedCheckbox][0]) + self.im_boxes[self.selectedCheckbox][2]
            self.im_boxes[self.selectedCheckbox][3] = -(y_pressed - self.im_boxes[self.selectedCheckbox][1]) + self.im_boxes[self.selectedCheckbox][3]
            self.im_boxes[self.selectedCheckbox][0] = x_pressed
            self.im_boxes[self.selectedCheckbox][1] = y_pressed
        elif (event.button() == Qt.RightButton): 
            x_pressed = round(event.pos().x() * self.ratio)
            y_pressed = round(event.pos().y() * self.ratio)
            self.im_boxes[self.selectedCheckbox][2] = x_pressed - self.im_boxes[self.selectedCheckbox][0]
            self.im_boxes[self.selectedCheckbox][3] = y_pressed - self.im_boxes[self.selectedCheckbox][1]
        self.fillTable()
        self.updateBoxesFunction()
            
    # Deletes an existing bounding box from the image
    def deleteLabelFunction(self): 
        alert = QMessageBox()
        alert.setIcon(QMessageBox.Warning)
        alert.setWindowTitle("Delete a Box")
        if self.selectedCheckbox == -1:
            alert.setStandardButtons(QMessageBox.Ok)
            alert.setText('You have not selected a box to delete. Please select and try again. Do not forget to press "Update" before trying to delete !')
            alert.exec()
        else:
            alert.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            alert.setText('Are you sure you want to delete this box ?')
            returnValue = alert.exec()
            
            if returnValue == QMessageBox.Yes:
                del self.classes_ids[self.selectedCheckbox]
                tmp = np.array([[0]])
                self.im_indices = np.delete(self.im_indices, self.selectedCheckbox,0)
                
                if not(self.segmFlag):
                    self.image = cv2.imread(self.loadedImage)
                    mask = np.zeros(self.image.shape[:2],np.uint8)        
                    bgdModel = np.zeros((1,65),np.float64)
                    fgdModel = np.zeros((1,65),np.float64)
                    box = self.im_boxes[self.selectedCheckbox]
                    rect = (int(box[0]),int(box[1]),int(box[2]),int(box[3])) 
                    cv2.grabCut(self.image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
                    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                    self.segmImage = self.segmImage -  self.image*mask2[:,:,np.newaxis]
                    for y in range(0, self.image.shape[0]):
                        for x in range(0, self.image.shape[1]):
                            if (self.segmImage[y][x][0] == 0) and (self.segmImage[y][x][1] == 0) and (self.segmImage[y][x][2] == 0):
                                self.segmImage[y][x][0] = 0
                                self.segmImage[y][x][1] = 0
                                self.segmImage[y][x][2] = self.gray[y][x]
                            elif (self.segmImage[y][x][0] == 255) and (self.segmImage[y][x][1] == 255) and (self.segmImage[y][x][2] == 255):
                                self.segmImage[y][x][0] = 0
                                self.segmImage[y][x][1] = 0
                                self.segmImage[y][x][2] = 255
                    self.image = self.segmImage
                    cv2.imwrite("segmented-image.jpg", self.segmImage)
                
                if len(self.im_indices) == 0:
                    tmp = np.array([])
                
                for i in range(1,len(self.im_indices)):
                    tmp = np.vstack((tmp,i))
                    
                self.updated = 1
                self.im_indices = tmp
                del self.im_boxes[self.selectedCheckbox][:]
                del self.im_boxes[self.selectedCheckbox]
                                
                self.updateImage()
    
    # Sets the buttons that the user can click on when he selects the bounding boxes option
    def boundingBoxesFunction(self):
        self.selectedCheckbox = -1
        
        if self.flag:
            self.flag = 0            
            self.updateBoxes.setEnabled(True)
            self.addLabel.setEnabled(True)
            self.moveLabel.setEnabled(True)
            self.adjustSizeButton.setEnabled(True)
            self.deleteLabel.setEnabled(True)            
        elif not(self.flag):
            self.flag = 1               
            self.updateBoxes.setEnabled(False)
            self.addLabel.setEnabled(False)
            self.moveLabel.setEnabled(False)
            self.adjustSizeButton.setEnabled(False)
            self.deleteLabel.setEnabled(False)            
        self.updateImage()
    
    # Sets the thickness of each bounding box based on his selection
    def updateBoxesFunction(self):        
        if self.segmentationFlag:
            self.image = self.segmImage
        else:
            if not(self.video):
                self.image = cv2.imread(self.loadedImage)
        size = max(self.pixmap.height(),self.pixmap.width())
        for i in range(1,self.rows):
            box = self.im_boxes[i-1]             
            label = (self.detTable.item(i,0).text())
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            if label in self.classes:                
                index = self.classes.index(label)  
                if self.detTable.cellWidget(i,1).isChecked():
                    self.selectedCheckbox = i-1
                    thickness = size / 571.4286
                    rect_size = 4
                    self.draw_bounding_box(self.image, label,index, x, y, (x+w), (y+h),thickness,rect_size)
                else:
                    rect_size = 2
                    thickness = size / 1000
                    self.draw_bounding_box(self.image, label,index, x, y, (x+w), (y+h),thickness,rect_size)
            elif label not in self.classes:
                index = -1
                if self.detTable.cellWidget(i,1).isChecked():
                    self.selectedCheckbox = i-1
                    thickness = size / 571.4286
                    rect_size = 4
                    self.draw_bounding_box(self.image, label,index, x, y, (x+w), (y+h),thickness,rect_size)
                    index = self.classes.index(label)
                else:
                    rect_size = 2
                    thickness = size / 1000
                    self.draw_bounding_box(self.image, label,index, x, y, (x+w), (y+h),thickness,rect_size)
                    index = self.classes.index(label)
            self.classes_ids[i-1] = index
        cv2.imwrite("object-detection.jpg", self.image)
        self.updated = 0
        self.label.setPixmap(QPixmap('object-detection.jpg'))
        
    # Quits the application
    def quitApp(self):
        QApplication.quit()
        
    # Saves the image and its bounding boxes into a temporary library. Saved in COCO format
    def saveToLibFunction(self):        
        if self.loadedImage in self.seen:
            print ("Already in library")
        else:
            self.loadSavedButton.setEnabled(True)
            self.seen.add(self.loadedImage)
            self.savedImages.append({'id':self.saveCounter, 'loaded_path': self.loadedImage, 'indices':self.im_indices,
                                    'boxes':self.im_boxes,'classes_ids':self.classes_ids})
            with open('bounding_boxes.csv', mode='a') as bounding_file:
                bounding_writer = csv.writer(bounding_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                bounding_writer.writerow([self.saveCounter, self.loadedImage, self.im_indices, self.im_boxes, self.classes_ids ])            
            self.image = cv2.imread(self.loadedImage)
            cv2.imwrite("object-detection.jpg", self.image)
            labelName = "label" + str(self.saveCounter)
            labelName = QLabel(self.window)
            labelName.setScaledContents(True)
            labelName.setGeometry(790 + 110*(self.savedPos%2), 50 + 100*int(self.savedPos/2), 90, 70)
            labelName.setPixmap(QPixmap('object-detection.jpg'))                     
            labelName.show()
            checkName = "chkBoxItem" + str(self.saveCounter)
            checkName = QCheckBox(self.window)
            self.savedBoxes.addButton(checkName)
            checkName.setCheckState(Qt.Unchecked)
            checkName.move(830 + 110*(self.savedPos%2),120 + 100*int(self.savedPos/2))
            checkName.show()
            self.savedBoxes.setId(checkName,self.saveCounter) 
            self.saveCounter += 1
            self.savedPos += 1
            if (self.savedPos == 12):
                self.savedPos = 0
                
    # Loads the image selected by the user and its bounding boxes and shows it in the GUI
    def loadSavedFunction(self):
        checkedBox = self.savedBoxes.checkedId()
        if checkedBox != (-1):
            self.image = cv2.imread(self.savedImages[checkedBox]['loaded_path'])
            self.pixmap = QPixmap(self.savedImages[checkedBox]['loaded_path'])        
            self.label.setScaledContents(True)
            self.label.setPixmap(self.pixmap)
            if (self.pixmap.width() > 660) or (self.pixmap.height() > 660):
                self.ratio = max(self.pixmap.width()/660,self.pixmap.height()/660) + 0.71
                self.label.resize(self.pixmap.width() / self.ratio,self.pixmap.height() / self.ratio )
                pixmapWidth = self.pixmap.width() / self.ratio
                pixmapHeight = self.pixmap.height() / self.ratio
            else:
                self.label.resize(self.pixmap.width(),self.pixmap.height())
                pixmapWidth = self.pixmap.width()
                pixmapHeight = self.pixmap.height()
                
            self.label.move(660/2-pixmapWidth/2,660/2-pixmapHeight/2)            
            cv2.imwrite("object-detection.jpg", self.image)
            
            self.loadedImage = self.savedImages[checkedBox]['loaded_path']
            self.classes_ids = self.savedImages[checkedBox]['classes_ids']
            self.im_indices = self.savedImages[checkedBox]['indices']
            self.im_boxes = self.savedImages[checkedBox]['boxes']              
            self.label.setPixmap(QPixmap('object-detection.jpg'))            
            self.segmImage = np.zeros((self.image.shape))        
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            self.flag = 1
            self.segmentationFlag = 0
            self.segmFlag = 1
            self.updateImage()
        
    # Loads a video selected by the user
    def loadVideoFunction(self):
        self.label.setText('Loading....')  
        
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        self.loadedVideo = QFileDialog.getOpenFileName(None,'Open file',desktop)[0]
        
        self.pixmap = QPixmap(self.loadedVideo)        
        self.label.setScaledContents(True)
        self.label.setPixmap(self.pixmap)
        self.boundingBoxes.setEnabled(True)
        self.originalImage.setEnabled(True)

        self.pauseButton = QPushButton(self.window)
        self.pauseButton.clicked.connect(self.pauseFunction)
        self.pauseButton.setGeometry(93, 565, 90, 30)
        self.pauseButton.setText("Pause")
        self.pauseButton.show()
        self.playButton = QPushButton(self.window)
        self.playButton.clicked.connect(self.playFunction)
        self.playButton.setGeometry(183, 565, 90, 30)
        self.playButton.setText("Play")
        self.playButton.show()
                
        self.pauseFlag = 0
        
        pixmapWidth = 660*0.71
        pixmapHeight = 660*0.71
        self.label.resize(pixmapWidth,pixmapHeight)
        self.label.move(660/2-pixmapWidth/2,660/2-pixmapHeight/2)
        
        self.video = 1
        self.flag = 1
        self.segmentationFlag = 0
        cap = cv2.VideoCapture(self.loadedVideo)
        while(cap.isOpened()):
            if (self.pauseFlag == 0):
                ret, frame = cap.read()
                if ret:
                    self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = self.image .shape 
                    self.blob = cv2.dnn.blobFromImage(self.image , self.scale, (416,416), (0,0,0), True, crop=False)
                    self.net.setInput(self.blob)
                    
                    self.yoloPredImage()
                    cv2.imwrite("object-detection.jpg", self.image)  
                    self.updateImage()
            
                    self.label.setPixmap(QPixmap('object-detection.jpg'))
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        cap.release()
        cv2.destroyAllWindows()
        self.video = 0
        
    def pauseFunction(self):
        self.pauseFlag = 1
        self.updateBoxes.setEnabled(True)
        self.addLabel.setEnabled(True)
        self.moveLabel.setEnabled(True)
        self.adjustSizeButton.setEnabled(True)
        self.deleteLabel.setEnabled(True)
        
    def playFunction(self):
        self.pauseFlag = 0
        self.updateBoxes.setEnabled(False)
        self.addLabel.setEnabled(False)
        self.moveLabel.setEnabled(False)
        self.adjustSizeButton.setEnabled(False)
        self.deleteLabel.setEnabled(False)
          
    # Loads an image based on the user's selection
    def loadImageFunction(self):
        self.label.setText('Loading....')  
        
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        self.loadedImage = QFileDialog.getOpenFileName(None,'Open file',desktop)[0]
        
        self.pixmap = QPixmap(self.loadedImage)        
        self.label.setScaledContents(True)
        self.label.setPixmap(self.pixmap)
        
        self.segmentationFlag = 0
        self.flag= 1
        
        self.image = cv2.imread(self.loadedImage)                
        self.segmImage = np.zeros((self.image.shape))        
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        if (self.pixmap.width() > 660) or (self.pixmap.height() > 660):
            self.ratio = max(self.pixmap.width()/660,self.pixmap.height()/660) + 0.71
            self.label.resize(self.pixmap.width() / self.ratio,self.pixmap.height() / self.ratio )
            pixmapWidth = self.pixmap.width() / self.ratio
            pixmapHeight = self.pixmap.height() / self.ratio
        else:
            self.ratio = 1
            self.label.resize(self.pixmap.width(),self.pixmap.height())
            pixmapWidth = self.pixmap.width()
            pixmapHeight = self.pixmap.height()

        self.label.move(660/2-pixmapWidth/2,660/2-pixmapHeight/2)
        self.blob = cv2.dnn.blobFromImage(self.image, self.scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(self.blob)
        
        self.yoloPredImage()
        
        self.showLibButton.setEnabled(True)        
        self.boundingBoxes.setEnabled(True)
        self.segmentationButton.setEnabled(True)
        self.originalImage.setEnabled(True)
        self.segmFlag = 1
        cv2.imwrite("object-detection.jpg", self.image)        
        self.label.setPixmap(QPixmap('object-detection.jpg'))
        self.emptyTable()
                    
    # Uses YOLO pre-trained CNN to identify objects into the image
    def yoloPredImage(self):
        outs = self.net.forward(self.get_output_layers())
        Width = self.image.shape[1]
        Height = self.image.shape[0]        

        class_ids = []
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
        
        self.classes_ids = class_ids
        self.im_indices = indices
        self.im_boxes = boxes    
        
    # Fills a table with the objects identified by YOLO and gives the ability to the user to select one of those
    def fillTable(self):     
        headers = ["Label", "Check"]
        self.col = len(headers)
        self.rows = len(self.classes_ids) + 1
        
        self.checkBoxes = QButtonGroup(self.detTable)
        self.checkBoxes.setExclusive(True)
        
        self.detTable.setRowCount(self.rows)
        self.detTable.setColumnCount(self.col)        
        
        for i in range(0,self.col):
            self.detTable.setItem(0,i, QTableWidgetItem(headers[i]))
        
        for i in range(1,self.rows):
            self.detTable.setItem(i,0, QTableWidgetItem(self.classes[self.classes_ids[i-1]]))
            name = "chkBoxItem" + str(i)
            name = QCheckBox()
            self.checkBoxes.addButton(name)
            name.setCheckState(Qt.Unchecked) 
            self.detTable.setCellWidget(i, 1, name)
            for j in range(1,self.col-1):
                self.detTable.setItem(i,j, QTableWidgetItem(str(round(self.im_boxes[i-1][j-1]))))            
            self.detTable.resizeColumnsToContents()
            
    # Empties the table filled with the objects identified by YOLO algorithm
    def emptyTable(self):
        self.col = 0
        self.rows = 0
        self.detTable.setRowCount(self.rows)
        self.detTable.setColumnCount(self.col)
            
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return self.output_layers   
    
    # Draws a bounding box for each identified object of the image
    def draw_bounding_box(self,img, label, class_id, x, y, x_plus_w, y_plus_h, thickness,rect_size):
        if class_id == -1:            
            color = np.random.uniform(0, 255, 3)
            choice = self.new_label_addition(label,color)
            if choice == 1:                    
                cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, rect_size)
                cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, thickness, color, 2)
            else:
                cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, rect_size) 
        else:
            color = self.COLORS[class_id]    
            cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, rect_size)
            cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, thickness, color, 2)
        
    # Warns the user that the label that he is about to add is not included in the dataset and asks him if he wants to add it
    def new_label_addition(self,label,color):
        alert = QMessageBox()
        alert.setIcon(QMessageBox.Information)
        alert.setWindowTitle("New label")
        alert.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        alert.setText('This label is not in the existing dataset, would you like to add it ?')
        returnValue = alert.exec()
        choice = 0
        if returnValue == QMessageBox.Yes:          
           self.classes.append(label)
           self.COLORS = np.vstack([self.COLORS, color])
           choice = 1
        return(choice)
       
    # Centers the GUI in the center of the screen
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    # Uses te GraphCut method to segment the image
    def segmentationFunction(self):
        self.image = cv2.imread(self.loadedImage)  
        mask = np.zeros(self.image.shape[:2],np.uint8)        
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        if self.segmFlag:  
            self.segmFlag = 0
            if(len(self.im_indices) != 0):
                for i in self.im_indices:
                    i = i[0]
                    box = self.im_boxes[i]
                    rect = (int(box[0]),int(box[1]),int(box[2]),int(box[3]))            
                    cv2.grabCut(self.image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
                    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
                    self.segmImage = self.segmImage +  self.image*mask2[:,:,np.newaxis]
    
            for y in range(0, self.image.shape[0]):
                for x in range(0, self.image.shape[1]):
                    if (self.segmImage[y][x][0] == self.segmImage[y][x][1]) and (self.segmImage[y][x][2] == self.segmImage[y][x][1]):
                        self.segmImage[y][x][0] = 0
                        self.segmImage[y][x][1] = 0
                        self.segmImage[y][x][2] = self.gray[y][x]
            cv2.imwrite("segmented-image.jpg", self.segmImage)        
        
        if not(self.segmentationFlag):
            self.segmentationFlag = 1
        else:
            self.segmentationFlag = 0
            
        self.updateImage()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
    
