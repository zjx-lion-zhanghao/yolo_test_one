import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
	model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')   # 修改yaml
	model.load('yolov8n.pt')  #加载预训练权重
	model.train(data='data.yaml',   #数据集yaml文件
	            imgsz=640,
	            epochs=200,
	            batch=16,
	            workers=32,
	            device='0',   #没显卡则将0修改为'cpu'
	            optimizer='SGD',
                amp = False,
	            cache=True,   #服务器可设置为True，训练速度变快
	)