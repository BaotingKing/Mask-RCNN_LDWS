"""
Mask R-CNN
Configurations and data loading code for Cityscapes.
------------------------------------------------------------

Usage: run from the command line as such:
     Please refer to code of coco
"""

import os
import sys
import time
import json
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
import zipfile
import urllib.request
import shutil

# Import Mask RCNN
ROOT_DIR = os.path.abspath("../../../")  # Root directory of the projectMi
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from demo.samples.Cityscaps import proc_cityscaps

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Training y sample category
############################################################
if False:
    CATEGORYS = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush']
else:
    # CATEGORYS = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'traffic light', 'stop sign']
    CATEGORYS = proc_cityscaps.std_class_big_cat


############################################################
#  Configurations
############################################################
class CityScapesConfig(Config):
    """Configuration for training on CityScapes.
    Derives from the base Config class and overrides values specific
    to the Cityscapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscape"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CATEGORYS)  # COCO has 80 classes

    STEPS_PER_EPOCH = 1000
    # IMAGE_MIN_DIM = 1024
    # IMAGE_MAX_DIM = 2048


############################################################
#  Dataset
############################################################
class CityScapesDataset(utils.Dataset):
    def __init__(self, class_map=None):
        super().__init__(class_map=None)
        # self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.dataset = dict()

    def load_cityscapes(self, dataset_dir, subset, class_ids=None,
                        class_map=None, return_cityscapes=False, auto_download=False):
        """Load a subset of the CityScapes dataset.
        dataset_dir: The root directory of the CityScapes dataset.
        subset: What to load (train, val, minival, valminusminival)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_cityscapes: If True, returns the CityScapes object.
        auto_download: Automatically download and unzip CityScapes images and annotations, but now is empty
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset)

        if subset in ['train', 'val']:
            print('loading annotations into memory...')
            tic = time.time()
            city_script_path = os.path.join(os.getcwd(), "cityscapes")
            annotation_file = os.path.join(os.path.join(city_script_path, subset), "label.json")
            if os.path.exists(annotation_file):
                dataset = json.load(open(annotation_file, 'r'))
                assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
                print('Done (t={:0.2f}s)'.format(time.time() - tic))
                self.dataset = dataset['labels']
        else:
            print('Can not load label files!')

        # Add classes
        for k, v in proc_cityscaps.class_label_id.items():
            self.add_class("cityscapes", v, k)

        # Add images
        for i in range(len(self.dataset)):
            img_info = self.dataset[i]
            img_path = proc_cityscaps.search_file(dataset_dir, img_info['img_name'])
            self.add_image(
                "cityscapes", image_id=i,
                path=img_path,
                width=img_info["width"],
                height=img_info["height"],
                annotations=img_info['object'])
        cityscapes = []
        if return_cityscapes:
            return cityscapes

    def auto_download(self, dataDir, dataType, dataYear):
        """Download dataset/annotations if requested."""
        print("This is empty and you can implement this function......")

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cityscapes image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cityscapes":
            return super(CityScapesDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "cityscapes.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CityScapesDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the cityscapes Website."""
        # TODO: not need
        pass

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  CityScapes Evaluation
############################################################
def build_cityscapes_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "cityscapes"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_cityscapes(model, dataset, cityscapes, eval_type="bbox", limit=0, image_ids=None):
    """Runs official CityScapes evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick CityScapes images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding CityScapes image IDs.
    cityscapes_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to CityScapes format
        # Cast masks to uint8 because cityscapes tools errors out on bool
        image_results = build_cityscapes_results(dataset, cityscapes_image_ids[i:i + 1],
                                                 r["rois"], r["class_ids"],
                                                 r["scores"],
                                                 r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    cityscapes_results = cityscapes.loadRes(results)

    # Evaluate
    cityscapesEval = COCOeval(cityscapes, cityscapes_results, eval_type)
    cityscapesEval.params.imgIds = cityscapes_image_ids
    cityscapesEval.evaluate()
    cityscapesEval.accumulate()
    cityscapesEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################
if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on CityScapes.')
    parser.add_argument("--command",
                        default="train",
                        metavar="<command>",
                        help="'train' or 'evaluate' on CityScapes")
    parser.add_argument('--dataset', required=False,
                        default="G://Dataset//Cityscape//",
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the CityScapes (default=2014)')
    parser.add_argument('--model', required=False,
                        default="F://projects//Mask_RCNN//mask_rcnn_coco.h5",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--schema', required=False,
                        default="coco",
                        metavar="<Schema>",
                        help="This is plane for train, e.g. coco/last/default, coco is fine_turn, last is continue")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip CityScapes files (default=False)',
                        type=bool)
    args = parser.parse_args()

    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CityScapesConfig()
    else:
        class InferenceConfig(CityScapesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    print('[Step 1]: Create model ..........................................')
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if "coco.h5" in args.model.lower():
        model_path = args.model
    elif "last.h5" in args.model.lower():
        # Find last trained weights
        model_path = model.find_last()
    elif "imagenet.h5" in args.model.lower():
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model    # This is default train on yourself model!

    # Load weights
    print("Loading weights ", model_path)
    init_with = args.schema
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on CityScapes, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    elif init_with == "default":
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CityScapesDataset()
        dataset_train.load_cityscapes(args.dataset, "train", auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CityScapesDataset()
        dataset_val.load_cityscapes(args.dataset, "val", auto_download=args.download)
        dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=1,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=2,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=2,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CityScapesDataset()
        val_type = "val" if args.year in '2017' else "minival"
        cityscapes = dataset_val.load_cityscapes(args.dataset, val_type, return_cityscapes=True,
                                                 auto_download=args.download)
        dataset_val.prepare()
        print("Running CityScapes evaluation on {} images.".format(args.limit))
        evaluate_cityscapes(model, dataset_val, cityscapes, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
