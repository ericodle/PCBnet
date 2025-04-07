import argparse
from utils.dataset import CocoDataset
import torch
from utils.model_utils import InferFasterRCNN, display_gt_pred
from pycocotools.coco import COCO
import os

def main(args):
    train_ds = CocoDataset(
        image_folder=args.image_folder,
        annotations_file=args.annotations_file,
        height=4000,
        width=6000
    )

    classnames = train_ds.get_classnames()
    print(classnames)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    IF_C = InferFasterRCNN(
        num_classes=train_ds.get_total_classes_count() + 1, 
        classnames=classnames
    )

    IF_C.load_model(checkpoint=args.checkpoint, device=device)

    cocoGt = COCO(args.annotations_file)
    imgIds = cocoGt.getImgIds() 
    print(len(imgIds))

    img_info = cocoGt.loadImgs(imgIds[args.image_id])[0]
    annIds = cocoGt.getAnnIds(imgIds=img_info['id'])
    ann_info = cocoGt.loadAnns(annIds)
    image_path = os.path.join(args.image_folder, img_info['file_name'])

    transform_info = CocoDataset.transform_image_for_inference(
        image_path,
        width=640,
        height=640
    )

    result = IF_C.infer_image(transform_info=transform_info, visualize=False)

    gts_cls = [i['category_id'] for i in ann_info]
    gts_bbox = [[
        i['bbox'][0],
        i['bbox'][1],
        i['bbox'][0] + i['bbox'][2],
        i['bbox'][1] + i['bbox'][3]
    ] for i in ann_info]  

    pred_boxes = result['unscaled_boxes']
    pred_classes = result['pred_classes']
    pred_scores = result['scores']

    display_gt_pred(
        image_path=image_path,
        gt_boxes=gts_bbox,
        gt_class=gts_cls,
        pred_boxes=pred_boxes,
        pred_class=pred_classes,
        pred_scores=pred_scores,
        classnames=classnames,  
        box_format='xyxy'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a COCO dataset image.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the COCO annotations file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--image_id", type=int, default=0, help="ID of the image to run inference on.")

    args = parser.parse_args()
    main(args)
