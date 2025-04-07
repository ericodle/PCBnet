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

    # Draw GT boxes (green)
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main(args):
    # Load dataset and COCO annotations
    train_ds = CocoDataset(
        image_folder=args.image_folder,
        annotations_file=args.annotations_file,
        height=4000,
        width=6000
    )
    classnames = train_ds.get_classnames()
    print(f"Detected classes: {classnames}")

    cocoGt = COCO(args.annotations_file)
    img_ids = cocoGt.getImgIds()
    print(f"Found {len(img_ids)} annotated images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InferFasterRCNN(
        num_classes=train_ds.get_total_classes_count() + 1,
        classnames=classnames
    )
    model.load_model(checkpoint=args.checkpoint, device=device)

    for img_id in img_ids:
        img_info = cocoGt.loadImgs(img_id)[0]
        image_path = os.path.join(args.image_folder, img_info['file_name'])

        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        ann_ids = cocoGt.getAnnIds(imgIds=img_info['id'])
        ann_info = cocoGt.loadAnns(ann_ids)

        # Prepare image for inference
        transform_info = CocoDataset.transform_image_for_inference(
            image_path,
            width=640,
            height=640
        )

        result = model.infer_image(transform_info=transform_info, visualize=False)

        # Ground truths
        gts_cls = [ann['category_id'] for ann in ann_info]
        gts_bbox = [[
            ann['bbox'][0],
            ann['bbox'][1],
            ann['bbox'][0] + ann['bbox'][2],
            ann['bbox'][1] + ann['bbox'][3]
        ] for ann in ann_info]

        # Predictions
        pred_boxes = result['unscaled_boxes']
        pred_classes = result['pred_classes']
        pred_scores = result['scores']

        # Output path
        save_path = None
        if args.output_folder:
            output_filename = os.path.splitext(img_info['file_name'])[0] + "_pred.png"
            save_path = os.path.join(args.output_folder, output_filename)

        display_gt_pred(
            image_path=image_path,
            gt_boxes=gts_bbox,
            gt_class=gts_cls,
            pred_boxes=pred_boxes,
            pred_class=pred_classes,
            pred_scores=pred_scores,
            classnames=classnames,
            box_format='xyxy',
            save_path=save_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a folder of COCO images.")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing images.")
    parser.add_argument("--annotations_file", type=str, required=True, help="Path to the COCO annotations JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output_folder", type=str, default=None, help="Optional folder to save visualized results.")

    args = parser.parse_args()
    main(args)
