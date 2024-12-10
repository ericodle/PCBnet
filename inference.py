from utils.dataset import CocoDataset
import torch
from utils.model_utils import InferFasterRCNN,display_gt_pred
from pycocotools.coco import COCO
import os


train_ds = CocoDataset(image_folder=r'/home/eo/FasterRCNN-Torchvision-FineTuning/imgs',
                  annotations_file=r'/home/eo/FasterRCNN-Torchvision-FineTuning/annotations/DSC01113.json',
                  height=4000,width=6000)

print(train_ds.get_classnames())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

IF_C = InferFasterRCNN(num_classes=train_ds.get_total_classes_count() + 1,
                       classnames=train_ds.get_classnames())

IF_C.load_model(checkpoint=r'/home/eo/FasterRCNN-Torchvision-FineTuning/exp/summary/10-12-2024-14-36-45/best_model.pth',
                device=device)

train_image_dir = r'/home/eo/FasterRCNN-Torchvision-FineTuning/imgs'
cocoGt=COCO(r"/home/eo/FasterRCNN-Torchvision-FineTuning/annotations/DSC01113.json")

imgIds = cocoGt.getImgIds() # all image ids
print(len(imgIds))

id = 0

img_info = cocoGt.loadImgs(imgIds[id])[0]
annIds = cocoGt.getAnnIds(imgIds=img_info['id'])
ann_info = cocoGt.loadAnns(annIds)
image_path = os.path.join(train_image_dir, 
                          img_info['file_name'])

transform_info = CocoDataset.transform_image_for_inference(image_path,width=640,height=640)

result = IF_C.infer_image(transform_info=transform_info,
                         visualize=False)
