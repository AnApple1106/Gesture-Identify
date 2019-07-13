import tkinter as tk 
from PIL import Image,ImageTk
from imutils.video import VideoStream
from selfTrain import *
from collectStaticData import *
from runner import *
import imutils
import cv2
import threading
import argparse
import time
import numpy as np 
import random

img = cv2.imread('testtest.jpg', 0)
img2 = cv2.imread('timg.jpg', 0)
source = []
source.append(img)
source.append(img2)

class App:
	def __init__(self, root):

		self.root = root
		self.panel = None	#画布空间内层放置图像的容器
		self.frames = None
		self.stopEvent = None	#窗口关闭事件
		self.closeflag = True	#摄像头循环的标志
		self.flags = False	#采集任务标志
		self.thread = None	#图像显示函数以子线程的方式运行
		self.vs = None		#图像帧变量
		self.root.title('珞樱V1.0')
		self.root.resizable(False, False)	
		#self.root.iconbitmap('logo.ico')
		self.mainframe = tk.LabelFrame(self.root, text='数据构建中...')
		self.mainframe.pack(padx=5, pady=10)

		self.frame = tk.LabelFrame(self.mainframe, text='状态')
		self.frame.pack(side=tk.RIGHT, padx=5, pady=5)

		self.canv = tk.Canvas(self.mainframe)
		self.canv.configure(background='black', relief='ridge', width=400, height=300, highlightthickness=5, borderwidth=5)

		#self.identifyShow = tk.Canvas(self.canv)
		#self.identifyShow.configure(background='black', width=160, height=120)
		#img = self.show_Image(source)
		#self.canv.create_image(0, 0, image=img)
		self.canv.pack(side=tk.LEFT, padx=5, pady=5)
		#姿势编号子控件窗口
		self.labelGestFrame = tk.Frame(self.frame)
		self.labelGestFrame.pack(side=tk.TOP, padx=5, pady=2)
		self.labelGest = tk.Label(self.labelGestFrame, text='姿势编号:')
		self.labelGest.pack(side=tk.LEFT, padx=5, pady=2)
		self.labelGestEntry = tk.Entry(self.labelGestFrame, width=10)
		self.labelGestEntry.pack(side=tk.LEFT, padx=5, pady=2)
		#当前姿势显示子控件
		self.labelGestShowFrame = tk.Frame(self.frame)
		self.labelGestShowFrame.pack(side=tk.TOP, padx=5, pady=2)
		self.labelGestShow = tk.Label(self.labelGestShowFrame, text='当前姿势为:')
		self.labelGestShow.pack(side=tk.LEFT, pady=2)
		self.labelGestShowInfo = tk.Label(self.labelGestShowFrame, text='     ')
		self.labelGestShowInfo.pack(side=tk.RIGHT, padx=5, pady=2)
		#起始索引子控件
		self.labelIndexFrame = tk.Frame(self.frame)
		self.labelIndexFrame.pack(side=tk.TOP, padx=5, pady=2)
		self.labelIndex = tk.Label(self.labelIndexFrame, text='起始索引:')
		self.labelIndex.pack(side=tk.LEFT, padx=5, pady=2)
		self.labelIndexEntry = tk.Entry(self.labelIndexFrame, width=10)
		self.labelIndexEntry.pack(side=tk.LEFT, padx=5, pady=2)
		#蓝牙链接状态子控件
		self.bluetoothFrame = tk.Frame(self.frame)
		self.bluetoothFrame.pack(side=tk.TOP, padx=5, pady=3)
		self.bluetoothLabel = tk.Label(self.bluetoothFrame, text='蓝牙连接状态:')
		self.bluetoothLabel.pack(side=tk.LEFT, padx=5, pady=2)
		self.bluetoothStatus = tk.Label(self.bluetoothFrame, text='连接状态')
		self.bluetoothStatus.pack(side=tk.LEFT, padx=5, pady=2)
		#按钮区域子控件
		self.buttonFrame = tk.Frame(self.frame)
		self.buttonFrame.pack(side=tk.BOTTOM, padx=5, pady=2)
		#收集按钮
		self.collectFrame = tk.LabelFrame(self.frame)
		self.collectFrame.pack(side=tk.TOP, padx=5, pady=2)
		self.startButton = tk.Button(self.collectFrame, text='开始收集', command=self.collect)
		self.startButton.pack(side=tk.LEFT, padx=5, pady=2)
		self.stopButton = tk.Button(self.collectFrame, text='停止收集', command=self.endCollect)
		self.stopButton.pack(side=tk.RIGHT, padx=5, pady=2)
		#开始训练按钮
		self.trainButton = tk.Button(self.buttonFrame, text='开始训练', command=self.Train)
		self.trainButton.pack(side=tk.TOP, padx=5, pady=2)
		#验证模块
		self.confirmFrame = tk.LabelFrame(self.buttonFrame)
		self.confirmFrame.pack(side=tk.TOP, padx=5, pady=2)
		self.confirmSeq = tk.Frame(self.confirmFrame)
		self.confirmSeq.pack(side=tk.TOP, padx=5, pady=2)
		self.confirmText = tk.Label(self.confirmSeq, text='验证序列')
		self.confirmText.pack(side=tk.LEFT, padx=5, pady=2)
		self.confirmEntry = tk.Entry(self.confirmSeq, width=10)
		self.confirmEntry.pack(side=tk.RIGHT, padx=5, pady=2)
		self.startConfirm = tk.Button(self.confirmFrame, text='开始验证', command=self.selfTrain)
		self.startConfirm.pack(side=tk.TOP, padx=5,pady=2)
		self.confirmResult = tk.Frame(self.confirmFrame)
		self.confirmResult.pack(side=tk.TOP, padx=5, pady=2)
		self.ResultText = tk.Label(self.confirmResult, text='验证结果')
		self.ResultText.pack(side=tk.LEFT, padx=5, pady=2)
		self.ResultStatus = tk.Label(self.confirmResult, text='占位符')
		self.ResultStatus.pack(side=tk.RIGHT, padx=5, pady=2)
		#识别按钮
		self.identify = tk.Button(self.buttonFrame, text='开始识别')
		self.identify.pack(side=tk.BOTTOM, padx=5, pady=2)
		

		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoloop,args=())	#为图像显示函数创建线程		
		#self.thread.setDaemon(True)
		self.thread.start()	#启动线程

		#self.root.wm_title("Window test")	
		self.root.wm_protocol("WM_DELETE_WINDOW",self.onClose)	#捕获窗口关闭行为，以ONCLOSE函数代替

	def show_Image(self, imgArray):
		src = random.choice(imgArray)
		dst = Image.fromarray(src)
		img = ImageTk.PhotoImage(dst)
		return img
		self.canv.create_image(0, 0, image=img)
		self.root.update_idletasks()

	def videoloop(self):					#视频显示函数
		print("[INFO] warming up WebCamera...")
		self.vs = VideoStream(0).start()		#从网络摄像头捕获视频流
		time.sleep(1.0)					#设定摄像头启动时间
		
		try:
			while self.closeflag:
				time.sleep(0.04)
				self.frames = self.vs.read()	#读取网络摄像头中的帧数据
				self.frame = imutils.resize(self.frames,width=400,height=300)	#对帧数据进行尺寸调整
				#print(self.frame.size)
				image = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)	#转换图像色彩模式以numpy_ndarray形式储存
				image = Image.fromarray(image).transpose(0)	#对图像进行水平反转（0为水平反转，1为垂直反转）
				image = ImageTk.PhotoImage(image)	#将RGB模式的IMAGE对象转换成图形

				if self.panel is None:			#创建并打包显示图片的容器
					self.panel = tk.Label(self.canv,image=image)
					self.panel.image = image
					self.panel.pack(side="left",padx=5,pady=5)
				else:
					self.panel.configure(image=image, highlightthickness=5, borderwidth=5)
					self.panel.image = image

		except RuntimeError as e:
			print("[INFO] caught a RuntimeError!")
			self.vs.stop()
		print("[INFO] WebCamera has been shut down!")

	def collect(self):
		try:
			Gestnum = int(self.labelGestEntry.get())
			Gestindex = int(self.labelIndexEntry.get())
			print(type(Gestindex))
		except ValueError as e:
			print("Please input a number!")
		result = collectStaticData1(self, Gestnum, Gestindex)
		self.labelGestShowInfo.configure(text=result[1])
		self.thread2 = threading.Thread(target=collectStaticData2, args=(self, result, Gestindex))
		self.flags = True
		self.thread2.start()
		#collectStaticData2(self)
	
	def endCollect(self):
		self.flags = False
		cv2.destroyAllWindows()
		print("停止收集")

	def Train(self):
		self.thread3 = threading.Thread(target=train, args=())
		self.thread3.start()
	
	def selfTrain(self):
		print("开始验证")
		Seq = int(self.confirmEntry.get())
		t = threading.Thread(target=self_Train, args=(self.frames, Seq))
		t.start()
		#self_Train(self.frames, Seq)
		
	def onClose(self):				#窗口关闭时的行为	
		self.vs.stop()
		self.closeflag = False
		self.root.destroy()
		self.root.quit()
if __name__ == "__main__":				
	root = tk.Tk()				#创建根窗口

	app = App(root)

	app.root.mainloop()
