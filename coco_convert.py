import os
import sys
import argparse
import labelme2coco

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert LabelMe annotations to COCO format.")
    parser.add_argument("annotation_dir", type=str, help="Path to the directory containing LabelMe annotations and images.")
    parser.add_argument("--train_split_rate", type=float, default=1, help="Train split rate (default: 0.85).")
    parser.add_argument("--category_id_start", type=int, default=1, help="Starting value for category IDs (default: 1).")

    args = parser.parse_args()

    annotation_dir = args.annotation_dir
    train_split_rate = args.train_split_rate
    category_id_start = args.category_id_start

    # Validate annotation directory
    if not os.path.isdir(annotation_dir):
        print(f"Error: Directory '{annotation_dir}' does not exist.")
        sys.exit(1)

    # Convert annotations
    try:
        labelme2coco.convert(
            labelme_folder=annotation_dir,
            export_dir=annotation_dir,  # Output directory is the same as the input
            train_split_rate=train_split_rate,
            category_id_start=category_id_start
        )
        print(f"Successfully converted LabelMe annotations to COCO format in '{annotation_dir}'.")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example execution:
# pip3 install labelme2coco
# python3 coco_convert.py ./annotations
# python3 coco_convert.py ./val_annotations