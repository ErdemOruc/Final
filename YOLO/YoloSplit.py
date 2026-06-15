import os
import random
import shutil

source_dir = "C:/Users/Erdem/Desktop/Final/YOLO/New folder" 
target_dir = "C:/Users/Erdem/Desktop/Final/Yolo/Leaf_Fruit"

rate = 0.8  
dirs = ["images/train", "images/val", "labels/train", "labels/val"]

for d in dirs:
    path = os.path.join(target_dir, d)
    os.makedirs(path, exist_ok=True)

all_pics = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.jpg'.upper()))]
random.shuffle(all_pics)

thepoint = int(len(all_pics) * rate)
train_pics = all_pics[:thepoint]
val_pics = all_pics[thepoint:]

def copy_dic(pics_list, category):
    for pic_name in pics_list:
        dic_name, _ = os.path.splitext(pic_name)
        txt_name = dic_name + ".txt"
        
        pic_source_path = os.path.join(source_dir, pic_name)
        txt_source_path = os.path.join(source_dir, txt_name)
        
        shutil.copy(pic_source_path, os.path.join(target_dir, "images", category, pic_name))
        
        target_txt_path = os.path.join(target_dir, "labels", category, txt_name)
        
        if os.path.exists(txt_source_path):
            shutil.copy(txt_source_path, target_txt_path)
        else:
            with open(target_txt_path, 'w') as f:
                pass 

copy_dic(train_pics, "train")
copy_dic(val_pics, "val")

print("\nDone")