#!/usr/bin/python
#from utilstest import UtilsTest#, getLogger
#logger = getLogger(__file__)
import sift_pyocl as sift
import numpy
import scipy.misc

import sys, os, copy, time
#import numpy, scipy.misc
from PyQt4 import QtGui, QtCore


#TODO: remove this after integrating in sift
class Enum(dict):
    """
    Simple class half way between a dict and a class, behaving as an enum
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError

'''
try:
    import sift
except:
    print("FATAL : sift module not found !")


try:
    from sift import param
except:
    print("WARNING: Parameters file not found ! Taking default parameters")
    par = Enum(DoubleImSize=0,
    InitSigma=1.6,
    BorderDist=5,
    Scales=3,
    PeakThresh=255.0 * 0.04 / 3.0,
    EdgeThresh=0.06,
    EdgeThresh1=0.08,
    OriSigma=1.5,
    MatchRatio=0.73)
'''



class MainWin(QtGui.QMainWindow):
    
    def __init__(self):
        super(MainWin, self).__init__()
        self.step = 0
        self.folder = ""
        self.MAX_KP = 5000 #TODO: user-defined
        '''
        self.DoubleImSize
        self.InitSigma
        self.BorderDist
        self.Scales
        self.PeakThresh
        self.EdgeThresh
        self.EdgeThresh1
        self.OriSigma
        self.MatchRatio
        '''
        self.setDefaultPars()
        
        
        self.initUI()
        
    def initUI(self):      

        btn1 = QtGui.QPushButton("Select a directory", self)
        btn1.move(30, 50)
        btn2 = QtGui.QPushButton("Start alignment", self)
        btn2.move(120, 170)
        btn1.clicked.connect(self.buttonClicked)            
        btn2.clicked.connect(self.buttonClicked)
        btn1.adjustSize()
        btn2.adjustSize()
        
        self.pbar = QtGui.QProgressBar(self)
        self.pbar.setGeometry(30, 40, 300, 25)
        self.pbar.move(30,210)
        
        lbl0 = QtGui.QLabel(self)
        text0 = lbl0.setText("Please select a folder containing the images to align,\n and press \"start alignment\".")
        lbl0.move(30, 10)
        lbl0.adjustSize() 
        self.lbl1 = QtGui.QLabel(self)
        self.lbl1.setText("No folder selected yet")
        self.lbl1.move(35,85); self.lbl1.adjustSize()
        
        lbl2 = QtGui.QLabel(self)
        text2 = lbl2.setText("Run on device : ")
        lbl2.move(30, 110)
        self.combo = QtGui.QComboBox(self)
        self.combo.move(145, 110)
        self.detectDevices()


        
        
        
        
        
        
        

#        self.btn3 = QtGui.QPushButton("Edit SIFT parameters", self)
#        self.btn3.move(230, 50)
#        self.btn3.clicked.connect(self.buttonClicked)
#        self.btn3.adjustSize()

        self.text_DoubleImSize = QtGui.QLabel(self); self.text_DoubleImSize.setText("DoubleImSize :")
        self.text_DoubleImSize.move(400,20)
        self.edit_DoubleImSize=QtGui.QLineEdit(self); self.edit_DoubleImSize.setText(str(self.DoubleImSize))
        self.edit_DoubleImSize.move(500,25)
        self.edit_DoubleImSize.adjustSize()
#        self.edit_DoubleImSize.hide(); self.text_DoubleImSize.hide()
        
        self.text_InitSigma = QtGui.QLabel(self); self.text_InitSigma.setText("InitSigma :")
        self.text_InitSigma.move(400,50)
        self.edit_InitSigma=QtGui.QLineEdit(self); self.edit_InitSigma.setText(str(self.InitSigma))
        self.edit_InitSigma.move(500,55)
        self.edit_InitSigma.adjustSize()
#        self.edit_InitSigma.hide(); self.text_InitSigma.hide()
        
        self.text_BorderDist = QtGui.QLabel(self); self.text_BorderDist.setText("BorderDist :")
        self.text_BorderDist.move(400,80)
        self.edit_BorderDist=QtGui.QLineEdit(self); self.edit_BorderDist.setText(str(self.BorderDist))
        self.edit_BorderDist.move(500,85)
        self.edit_BorderDist.adjustSize()
#        self.edit_BorderDist.hide(); self.text_BorderDist.hide()
        
        self.text_Scales = QtGui.QLabel(self); self.text_Scales.setText("Scales :")
        self.text_Scales.move(400,110)
        self.edit_Scales=QtGui.QLineEdit(self); self.edit_Scales.setText(str(self.Scales))
        self.edit_Scales.move(500,115)
        self.edit_Scales.adjustSize()
#        self.edit_Scales.hide(); self.text_Scales.hide()
        
        self.text_PeakThresh = QtGui.QLabel(self); self.text_PeakThresh.setText("PeakThresh :")
        self.text_PeakThresh.move(400,140)
        self.edit_PeakThresh=QtGui.QLineEdit(self); self.edit_PeakThresh.setText(str(self.PeakThresh))
        self.edit_PeakThresh.move(500,145)
        self.edit_PeakThresh.adjustSize()
#        self.edit_PeakThresh.hide(); self.text_PeakThresh.hide()
        
        self.text_EdgeThresh = QtGui.QLabel(self); self.text_EdgeThresh.setText("EdgeThresh :")
        self.text_EdgeThresh.move(400,175)
        self.edit_EdgeThresh=QtGui.QLineEdit(self); self.edit_EdgeThresh.setText(str(self.EdgeThresh))
        self.edit_EdgeThresh.move(500,180)
        self.edit_EdgeThresh.adjustSize()
#        self.edit_EdgeThresh.hide(); self.text_EdgeThresh.hide()
        
        self.text_EdgeThresh1 = QtGui.QLabel(self); self.text_EdgeThresh1.setText("EdgeThresh1 :")
        self.text_EdgeThresh1.move(400,205)
        self.edit_EdgeThresh1=QtGui.QLineEdit(self); self.edit_EdgeThresh1.setText(str(self.EdgeThresh1))
        self.edit_EdgeThresh1.move(500,210)
        self.edit_EdgeThresh1.adjustSize()
#        self.edit_EdgeThresh1.hide(); self.text_EdgeThresh1.hide();
        
        self.text_OriSigma = QtGui.QLabel(self); self.text_OriSigma.setText("OriSigma :")
        self.text_OriSigma.move(400,235)
        self.edit_OriSigma=QtGui.QLineEdit(self); self.edit_OriSigma.setText(str(self.OriSigma))
        self.edit_OriSigma.move(500,240)
        self.edit_OriSigma.adjustSize()
#        self.edit_OriSigma.hide(); self.text_OriSigma.hide()
        
        self.text_MatchRatio = QtGui.QLabel(self); self.text_MatchRatio.setText("Match Ratio :")
        self.text_MatchRatio.move(400,265)
        self.edit_MatchRatio=QtGui.QLineEdit(self); self.edit_MatchRatio.setText(str(self.MatchRatio))
        self.edit_MatchRatio.move(500,270)
        self.edit_MatchRatio.adjustSize()
#        self.edit_MatchRatio.hide(); self.text_MatchRatio.hide()
        
        btn4 = QtGui.QPushButton("Reset default parameters", self)
        btn4.move(450, 295)
        btn4.clicked.connect(self.buttonClicked)
        btn4.adjustSize()
        
        
        
        self.statusBar()
        self.setGeometry(300, 300, 650, 350)
        self.setWindowTitle('SIFT main window')
        self.show()



        
    def buttonClicked(self):
        stext = self.sender().text()
        if stext == "Select a directory":
            dname = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
            self.lbl1.setText(dname)
            self.lbl1.adjustSize()
            self.folder = dname
        if stext == "Start alignment":
            self.step = self.step + 1
            self.pbar.setValue(self.step)
            self.runSift()
            
        '''
        if stext == "Edit SIFT parameters":
            self.edit_DoubleImSize.show(); self.text_DoubleImSize.show()
            self.edit_InitSigma.show(); self.text_InitSigma.show()
            self.edit_BorderDist.show(); self.text_BorderDist.show()
            self.edit_Scales.show(); self.text_Scales.show()
            self.edit_PeakThresh.show(); self.text_PeakThresh.show()
            self.edit_EdgeThresh.show(); self.text_EdgeThresh.show()
            self.edit_EdgeThresh1.show(); self.text_EdgeThresh1.show();
            self.edit_OriSigma.show(); self.text_OriSigma.show()
            self.edit_MatchRatio.show(); self.text_MatchRatio.show()
            self.btn3.setText("Hide SIFT parameters")
        if stext == "Hide SIFT parameters":  
            self.edit_DoubleImSize.hide(); self.text_DoubleImSize.hide()
            self.edit_InitSigma.hide(); self.text_InitSigma.hide()
            self.edit_BorderDist.hide(); self.text_BorderDist.hide()
            self.edit_Scales.hide(); self.text_Scales.hide()
            self.edit_PeakThresh.hide(); self.text_PeakThresh.hide()
            self.edit_EdgeThresh.hide(); self.text_EdgeThresh.hide()
            self.edit_EdgeThresh1.hide(); self.text_EdgeThresh1.hide();
            self.edit_OriSigma.hide(); self.text_OriSigma.hide()
            self.edit_MatchRatio.hide(); self.text_MatchRatio.hide()
            self.btn3.setText("Edit SIFT parameters")
        '''



    def runSift(self):
    
        if not(os.path.isdir(self.folder)):
            r = QtGui.QMessageBox.question(self, 'Error', "Please choose a folder first", QtGui.QMessageBox.Ok, QtGui.QMessageBox.Ok)
        else:
            rep = self.folder
            files = [ os.path.join(rep,f) for f in os.listdir(rep) if os.path.isfile(os.path.join(rep,f)) ]
            nbfiles = len(files)
            i=0
            dtype_kp = numpy.dtype([('x', numpy.float32),
                                    ('y', numpy.float32),
                                    ('scale', numpy.float32),
                                    ('angle', numpy.float32),
                                    ('desc', (numpy.uint8, 128))
                                    ])
#            keypoints_global = numpy.recarray(shape=(nbfiles,self.MAX_KP), dtype=dtype_kp)
#            match_global = numpy.recarray(shape=(nbfiles, self.MAX_KP, 2), dtype=dtype_kp)
            
            t0 = time.time()
            for filename in files[:]:
                '''
                Step 1 : keypoints computation
                '''
                self.statusBar().showMessage('[Step 1] Processing image ' + os.path.basename(filename))
                image = scipy.misc.imread(filename)
                plan = sift.SiftPlan(template=image, devicetype="gpu") #TODO: devicetype=..., device=device #user-defined
                kp = plan.keypoints(image) #TODO : pipeline of frames, instead of re-launching plan
                sh = kp.shape[0]
#                keypoints_global[i,sh] = kp
                if (i == 0): ref = copy.deepcopy(kp) #alignment is done on the first image #TODO
                i+=1
                self.statusBar().showMessage('[Step 1] Processing frame ' + os.path.basename(filename) + ' - ' + str(sh) + ' keypoints found')
                
                '''
                Step 2 : image matching
                '''
                if (i != 0): #alignment is done on the first image #TODO
                    mp = sift.MatchPlan() #TODO : pipeline of frames, instead of re-launching matchplan
                    m = mp.match(kp, ref)
                    self.statusBar().showMessage('[Step 2] Processing frame ' + os.path.basename(filename) + ' - ' + str(m.shape[0]) + ' match found')
                
                self.step = self.step + 100.0/nbfiles
                self.pbar.setValue(self.step)
            t1 = time.time()
            self.statusBar().showMessage('Process took ' + str(t1-t0)[0:4] + ' s')


    
    def setDefaultPars(self):
        self.DoubleImSize=0
        self.InitSigma=1.6
        self.BorderDist=5
        self.Scales=3
        self.PeakThresh=255.0 * 0.04 / 3.0
        self.EdgeThresh=0.06
        self.EdgeThresh1=0.08
        self.OriSigma=1.5
        self.MatchRatio=0.73


    def detectDevices(self):
    
        #TODO : import .opencl then list the devices
        self.combo.addItem("GPU")
        self.combo.activated[str].connect(self.onActivated) 
    
    
    def onActivated(self, text):
        #TODO : with @param text, select the platform and device
        self.statusBar().showMessage('Device selected : ' + text)
        
        
        








def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = MainWin()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    
    
    
    

