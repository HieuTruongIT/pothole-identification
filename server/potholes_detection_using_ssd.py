import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import cv2
from PIL import Image
import seaborn as sns
import copy
import torch
from torch.utils.data import Dataset,DataLoader,Subset
import torch.optim as optim
import torchvision
from torchvision.models.detection.ssd import SSDHead,det_utils
from torchvision.models.detection import ssd300_vgg16,SSD300_VGG16_Weights
import torchvision.transforms.functional as tf
import albumentations as A
import pycocotools
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision
np.random.seed(42)
torch.manual_seed(42)
import sys
sys.stdout.reconfigure(encoding='utf-8')

img_dir= r"C:\Users\trong\Desktop\NoctisAI - Detection\dataset\Pothole Detection\images"
annot_dir= r"C:\Users\trong\Desktop\NoctisAI - Detection\dataset\Pothole Detection\annotations"

classes=["background","pothole"]

num_classes=2
device="cuda" if torch.cuda.is_available() else "cpu"
batch_size=4
epochs=40
learning_rate=3e-5

model_weights_file="model.pth"

threshold=0.15
iou_threshold=0.55
def parse_xml(annot_path):
    tree=ET.parse(annot_path)
    root=tree.getroot()
    
    width=int(root.find("size").find("width").text)
    height=int(root.find("size").find("height").text)
    boxes=[]
    
    for obj in root.findall("object"):
        bbox=obj.find("bndbox")
        xmin=int(bbox.find("xmin").text)
        ymin=int(bbox.find("ymin").text)
        xmax=int(bbox.find("xmax").text)
        ymax=int(bbox.find("ymax").text)
        
        boxes.append([xmin,ymin,xmax,ymax])
        
    return boxes,height,width
ignore_img=[]
for annot_name in os.listdir(annot_dir):
    img_name=annot_name[:-4]+".png"
    annot_path=os.path.join(annot_dir,annot_name)
    boxes,height,width=parse_xml(annot_path)
    
    for box in boxes:
        if box[0]<0 or box[0]>=box[2] or box[2]>width:
            print(box[0],box[2],width)
            print("x",annot_name)
            print("*"*50)
            ignore_img.append(img_name)
        elif box[1]<0 or box[1]>=box[3] or box[3]>height:
            print(box[1],box[3],height)
            print("y",file_name)
            print("*"*50)
            ignore_img.append(img_name)
ignore_img


train_transform=A.Compose([A.HorizontalFlip(),
                           A.ShiftScaleRotate(rotate_limit=15,value=0,
                                              border_mode=cv2.BORDER_CONSTANT),

                           A.OneOf(
                                   [A.CLAHE(),
                                    A.RandomBrightnessContrast(),
                                    A.HueSaturationValue()],p=1),
                           A.GaussNoise(),
                           A.RandomResizedCrop(height=480,width=480)],
                          bbox_params=A.BboxParams(format="pascal_voc",min_visibility=0.15,
                                                   label_fields=["labels"]))
                           
val_transform=A.Compose([A.Resize(height=480,width=480)],
                        bbox_params=A.BboxParams(format="pascal_voc",min_visibility=0.15,
                                                 label_fields=["labels"]))



class PotholeDetection(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None, specific_img_name=None):
        super().__init__()
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.img_list = sorted([img for img in os.listdir(self.img_dir) if img not in ignore_img])
        self.transform = transform
        
        if specific_img_name:
            self.img_list = [specific_img_name]
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        annot_name = img_name[:-4] + ".xml"
        annot_path = os.path.join(self.annot_dir, annot_name)
        boxes, height, width = parse_xml(annot_path)
        labels = [1] * len(boxes)
        
        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]
        
        if len(np.array(boxes).shape) != 2 or np.array(boxes).shape[-1] != 4:
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]
                
        img = img / 255
        img = tf.to_tensor(img)
        img = img.to(dtype=torch.float32)
        
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["id"] = torch.tensor(idx)
            
        return img, target

train_ds=PotholeDetection(img_dir,annot_dir,train_transform)
from torch.utils.data import Subset
def get_image_name_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.readline().strip()
    return None

img_list = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]  
file_path = r"C:\Users\trong\Desktop\NoctisAI - Detection\server\image-name-file.txt"
val_img_name = get_image_name_from_file(file_path)

if val_img_name:
    val_idx = [i for i, img_name in enumerate(img_list) if img_name == val_img_name]
    train_idx = [i for i in range(len(img_list)) if img_list[i] != val_img_name]

    train_ds = Subset(PotholeDetection(img_dir, annot_dir, train_transform), train_idx)
    val_ds = Subset(PotholeDetection(img_dir, annot_dir, val_transform), val_idx)

    print(f"Tập huấn luyện có {len(train_ds)} ảnh.")
    print(f"Tập kiểm thử có {len(val_ds)} ảnh.")

    if len(val_ds) > 0:
        test_img, test_target = val_ds[0]
        print(f"Tên ảnh trong tập kiểm thử: {img_list[val_idx[0]]}")
        test_img = test_img.permute(1, 2, 0).cpu().numpy()
        # plt.imshow(test_img)
        # plt.axis("off")
        # plt.title(f"Ảnh kiểm thử: {val_img_name}")
        # plt.show()
else:
    print("Không tìm thấy ảnh kiểm thử trong file image-name-file.txt.")

len(val_ds)

def show_bbox(img,target,color=(0,255,0)):
    img=np.transpose(img.cpu().numpy(),(1,2,0))
    boxes=target["boxes"].cpu().numpy().astype("int")
    labels=target["labels"].cpu().numpy()
    img=img.copy()
    for i,box in enumerate(boxes):
        idx=int(labels[i])
        text=classes[idx]

        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color,2)
        y=box[1]-10 if box[1]-10>10 else box[1]+10
        cv2.putText(img,text,(box[0],y),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        
    return img
# fig,axes=plt.subplots(4,2,figsize=(12,24))
# plt.subplots_adjust(wspace=0.1,hspace=0.1)
# ax=axes.flatten()

# idxs=np.random.choice(range(len(train_ds)),8)
# for i,idx in enumerate(idxs):
#     img, target = train_ds[0] 
#     # output_img=show_bbox(img,target)
#     # ax[i].imshow(output_img)
#     # ax[i].axis("off")

def collate_fn(batch):
    return tuple(zip(*batch))
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True if device=="cuda" else False)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True if device=="cuda" else False)

model=ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

in_channels=det_utils.retrieve_out_channels(model.backbone,(480,480))
num_anchors=model.anchor_generator.num_anchors_per_location()
model.head=SSDHead(in_channels=in_channels,num_anchors=num_anchors,
                   num_classes=num_classes)

model.to(device)
for params in model.backbone.features.parameters():
    params.requires_grad=False
    
parameters=[params for params in model.parameters() if params.requires_grad]

optimizer=optim.Adam(parameters,lr=learning_rate)
lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                  patience=7, threshold=0.0001)

def get_lr(optimizer):
    for params in optimizer.param_groups:
        return params["lr"]
    
model_weights_pth= r"C:\Users\trong\Desktop\NoctisAI - Detection\dataset\Pothole Detection Learned Weights\model.pth"
model=ssd300_vgg16()

in_channels=det_utils.retrieve_out_channels(model.backbone,(480,480))
num_anchors=model.anchor_generator.num_anchors_per_location()
model.head=SSDHead(in_channels=in_channels,num_anchors=num_anchors,
                   num_classes=num_classes)

model.load_state_dict(torch.load(model_weights_pth,map_location=device))
model.to(device)
def preprocess_bbox(prediction): 
    processed_bbox={}
    
    boxes=prediction["boxes"][prediction["scores"]>=threshold]
    scores=prediction["scores"][prediction["scores"]>=threshold]
    labels=prediction["labels"][prediction["scores"]>=threshold]
    nms=torchvision.ops.nms(boxes,scores,iou_threshold=iou_threshold)
            
    processed_bbox["boxes"]=boxes[nms]
    processed_bbox["scores"]=scores[nms]
    processed_bbox["labels"]=labels[nms]
    
    return processed_bbox
metric=MeanAveragePrecision(box_format='xyxy',class_metrics=True)
metric.to(device)

model.eval()
with torch.no_grad():
    for imgs,targets in val_dl:
        imgs=[img.to(device) for img in imgs]
        targets=[{k:v.to(device) for (k,v) in d.items()} for d in targets]
        predictions=model(imgs)
        
        results=[]
        for prediction in predictions:
            results.append(preprocess_bbox(prediction))
        
        metric.update(results,targets)
        
results=metric.compute()
mean_ap=results["map"].item()
mean_ap_50=results["map_50"].item()
mean_ap_75=results["map_75"].item()

print(f"Mean Average Precision[0.5:0.95:0.05] : {mean_ap:.4f}")
print(f"Mean Average Precision @ 0.5          : {mean_ap_50:.4f}")
print(f"Mean Average Precision @ 0.75         : {mean_ap_75:.4f}")


val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
imgs, targets = next(iter(val_dl))
model.eval()

with torch.no_grad():
    output = model([img.to(device) for img in imgs])

prediction = output[0]
predict = preprocess_bbox(prediction)

num_potholes = len(predict["boxes"])
total_area = 0
damaged_area = 0

image_width = imgs[0].shape[2]
image_height = imgs[0].shape[1]
road_area = image_width * image_height

for box in predict["boxes"]:
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    total_area += width * height
    damaged_area += width * height

damage_ratio = total_area / road_area if road_area > 0 else 0

output_dir = r"C:/Users/trong/Desktop/NoctisAI - Detection/server/Real-Time-Pothole-Detection/"
output_path = os.path.join(output_dir, "pothole_detection_single_image.png")

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
output_img = show_bbox(imgs[0], predict, color=(255, 0, 0))
ax.imshow(output_img)
ax.axis("off")

ax.set_title(f"Detected Potholes: {num_potholes}, Damage Ratio: {damage_ratio:.2f}")

plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
# plt.show()

