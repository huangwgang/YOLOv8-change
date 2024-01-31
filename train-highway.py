import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/lyz-change/yolov8-goldyolo-seg.yaml')
    model.load('yolov8n-seg.pt') # loading pretrain weights
    model.train(data='dataset/data-highway.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=8,
                close_mosaic=10,
                workers=4,
                device='cpu',
                optimizer='SGD', # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )