# PCBnet

## Setup

Download the repository:

```sh
git clone https://github.com/ericodle/PCBnet
```

...create a virtual environment:

```sh
python3 -m venv env
```

...activate the virtual environment:


```sh
source env/bin/activate
```

...and install the dependencies:

  ```sh
pip3 install -r requirements.txt
  ```

## Label PCB Components

We use [labelme](https://github.com/wkentaro/labelme) to draw boxes around our components of interest.

Labelme generates .JSON annotation files of the same filename as the source image.

## Organize Training Data

Training images should be placed in a singel folder.
For example, `./train_imgs`

Inside `./train_imgs`, there should also be `train.json` and `val.json`.
These files contain the annotation data for the training process.


## Convert Annotations to COCO Format

We use [labelme2coco](https://github.com/fcakyon/labelme2coco) to convert the annotations into COCO format.

```sh
pip3 install labelme2coco
```

and then 

```sh
python3 coco_convert.py ./train_labels
python3 coco_convert.py ./val_labels
```

## Train

```sh
python train.py \
        --epoch 10 \
        --train_image_dir ./train_imgs \
        --val_image_dir ./train_imgs \
        --train_coco_json ./train_imgs/train.json \
        --val_coco_json ./train_imgs/val.json \
        --batch_size 4 \
        --exp_folder ./
```

## Inference

```sh
python3 run_inference.py \
        --image_folder ./test_imgs \
        --annotations_file ./test_imgs/test.json \
        --checkpoint ./summary/<your-output-file>/best_model.pth \
        --image_id 0
```