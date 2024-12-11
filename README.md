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
PYTHON_VERSION=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
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

### Step 3: Convert Annotations to COCO Format

We use [labelme2coco](https://github.com/fcakyon/labelme2coco) to convert the annotations into COCO format.

### Step 4: Organize Training Data

Training images should be placed in a clearly-labeled folder.
For this example, we place them in ./imgs

Annotations should also be palced in a clearly-labeled folder.
Here, we place them in ./annotations

## Fine-Tune fasterrcnn_resnet50

### Step 1: Execute train.py

Below is an example command. 

Feel free to change filepaths and training parameters as needed.

```sh
python train.py \
    --epoch 10 \
    --train_image_dir /home/eo/FasterRCNN-Torchvision-FineTuning/imgs \
    --val_image_dir /home/eo/FasterRCNN-Torchvision-FineTuning/val_imgs \
    --train_coco_json /home/eo/FasterRCNN-Torchvision-FineTuning/annotations/DSC01113.json \
    --val_coco_json /home/eo/FasterRCNN-Torchvision-FineTuning/val_annotations/DSC01113.json \
    --batch_size 16 \
    --exp_folder /home/eo/FasterRCNN-Torchvision-FineTuning
```

### Step 2: View Fine-Tuning Metrics

We use tensorboard to catch potential issues with the training process (overfitting, etc.)

*Be sure to correct the filepath to your specific situation*

```sh
tensorboard --logdir=./exp/summary/<your-folder>/ --port=6006
```

Navigate to http://localhost:6006/ in your browser to see the graphs.


## Inference





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
