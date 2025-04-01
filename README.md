# PCBnet

## About this Project

coming soon
 
## Pre-requisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Check your current version like this:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

### Step 1: Download the repository

```sh
git clone https://github.com/ericodle/PCBnet
```

### Step 2: Create a virtual environment

```sh
python3 -m venv env
```

Then, activate the virtual environment.


```sh
source env/bin/activate
```

### Step 3: Install requirements.txt

  ```sh
pip3 install -r requirements.txt
  ```
### Step 4: Replace coco.py with patched version

I suspect labelme2coco does not do a perfect job putting the annotation data in the expected format.

For now, this patched version of coco.py fixes the issue.

```sh
PYTHON_VERSION=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
TARGET_DIR="./env/lib/$PYTHON_VERSION/site-packages/pycocotools/"

# Print the target directory
echo "Target Directory: $TARGET_DIR"

# Move the file
cp ./coco.py "$TARGET_DIR/coco.py"
```

## Prepare Training Data

### Step 1: Take photos of PCBs

📸 

### Step 2: Label PCB Components

We use [labelme](https://github.com/wkentaro/labelme) to draw boxes around our components of interest.

Labelme generates .JSON annotation files of the same filename as the source image.

### Step 3: Organize Training Data

Training images should be placed in their own folder.
For this example, we place them in `./train_imgs`

Label annotations should also be placed in their own folder.
Here, we place them in `./train_labels`

For validation images, create similar folders for `./val_imgs` and `./val_labels`

### Step 4: Convert Annotations to COCO Format

We use [labelme2coco](https://github.com/fcakyon/labelme2coco) to convert the annotations into COCO format.

```sh
pip3 install labelme2coco
```

and then 

```sh
python3 coco_convert.py ./train_labels
python3 coco_convert.py ./val_labels
```

## Fine-Tune fasterrcnn_resnet50

### Step 1: Execute train.py
```sh
python3 train.py
```

### Step 2: View Fine-Tuning Metrics

We use tensorboard to catch potential issues with the training process (overfitting, etc.)

*Be sure to correct the filepath to your specific situation*

```sh
tensorboard --logdir=./exp/summary/<your-folder>/ --port=6006
```

Navigate to http://localhost:6006/ in your browser to see the graphs.


## Train

```sh
python train.py \
        --epoch 10 \
        --train_image_dir ./train_imgs \
        --val_image_dir ./train_imgs \
        --train_coco_json ./train_imgs/train.json \
        --val_coco_json ./train_imgs/val.json \
        --batch_size 16 \
        --exp_folder ./
```

## Inference

```sh
python3 run_inference.py \
        --image_folder ./test_imgs \
        --annotations_file ./test_imgs/test.json \
        --checkpoint ./exp/summary/01-04-2025-16-48-15/best_model.pth \
        --image_id 0
```