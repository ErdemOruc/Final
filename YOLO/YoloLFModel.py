# pyrefly: ignore [missing-import]
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov10m.pt')
    
    results = model.train(
        data='C:/Users/Erdem/Desktop/Final/Yolo/fldata.yaml',
        epochs=150,                 
        imgsz=640,
        batch=-1,                  
        patience=30,                
        name='WeightForLeafOrFruit',
        dropout=0.1,              
        cls=0.5,
        mixup=0.1,
        close_mosaic=15,  
    
        device=0,         
        workers=2,
        cos_lr=True,      
        save_period=10,   
   
        degrees=15.0,    
        fliplr=0.5,      
        flipud=0.2,       
        scale=0.2,        
        hsv_h=0.015,     
        hsv_s=0.7,        
        hsv_v=0.4        
    )