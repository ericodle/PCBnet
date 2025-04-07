import argparse
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO

from utils.dataset import CocoDataset
from utils.model_utils import InferFasterRCNN


def display_gt_pred(
    image_path,
    gt_boxes,
    gt_class,
    pred_boxes,
    pred_class,
    pred_scores,
    classnames,
    box_format='xyxy',
    save_path=None
):
    """Displays ground truth and prediction boxes on the image."""
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Draw ground truth boxes (green)
    for bbox, cls in zip(gt_boxes, gt_class):
        x0, y0, x1, y1 = bbox
        width, height = x1 - x0, y1 - y0
        rect = patches.Rectangle((x0, y0), width, height, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0, y0, classnames[cls - 1], color='g', fontsize=10, verticalalignment='top')

    # Draw predicted boxes (red)
    for bbox, cls, score in zip(pred_boxes, pred_class, pred_scores):
        x0, y0, x1, y1 = bbox
        width, height = x1 - x0, y1 - y0
        rect = patches.Rectangle((x0, y0), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        label = f"{classnames[cls]}: {score:.2f}"
        ax.text(x0, y0, label, color='r', fontsize=10, verticalalignment='bottom')

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved annotated image to: {save_path}")
    else:
        plt.show()


def main(args):
    # Load dataset
    train_ds = CocoDataset(
        image_folder=args.image_folder,
        annotations_file=args.annotations_file,
        height=4000,
        width=6000
    )

    classnames = train_ds.get_classnames()
    print("Detected Classes:", classnames)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = InferFasterRCNN(
        num_classes=train_ds.get_total_classes_count() + 1,
        classnames=classnames
    )
    model.load_model(checkpoint=args.checkpoint, device=device)

    # Load image info and annotations from COCO
    cocoGt = COCO(args.annotations_file)
    imgIds = cocoGt.getImgIds()
    print(f"Total images in dataset: {len(imgIds)}")

    img_info = cocoGt.loadImgs(imgIds[args.image_id])[0]
    image_path = os.path.join(args.image_folder, img_info['file_name'])
    annIds = cocoGt.getAnnIds(imgIds=img_info['id'])
    ann_info = cocoGt.loadAnns(annIds)

    # Prepare image for inference
    transform_info = CocoDataset.transform_image_for_inference(
        image_path,
        width=640,
        height=640
    )

    # Run inference
    result = model.infer_image(transform_info=transform_info, visualize=False)

    # Ground truth boxes and class labels
    gts_cls = [ann['category_id'] for ann in ann_info]
    gts_bbox = [[
        ann['bbox'][0],
        ann['bbox'][1],
        ann['bbox'][0] + ann['bbox'][2],
        ann['bbox'][1] + ann['bbox'][3]
    ] for ann in ann_info]

    # Prediction results
    pred_boxes = result['unscaled_boxes']
    pred_classes = result['pred_classes']
    pred_scores = result['scores']

    # Visualize predictions
    display_gt_pred(
        image_path=image_path,
        gt_boxes=gts_bbox,
        gt_class=gts_cls,
        pred_boxes=pred_boxes,
        pred_class=pred_classes,
        pred_scores=pred_scores,
        classnames=classnames,
        box_format='xyxy',
        save_path=args.save_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a COCO dataset image.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the COCO annotations file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--image_id", type=int, default=0, help="ID of the image to run inference on.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save output image (optional).")

    args = parser.parse_args()
    main(args)
