import ssl
import glob
import os
from pathlib import Path
import time

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.detection import IntersectionOverUnion, MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm
from engine import evaluate
from Averager import Averager
from FracturedDataset import FracturedDataset
from pprint import pprint

BATCH_SIZE = 16  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 50  # number of epochs to train for
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
TRAIN_DIR = '\\train'
# validation images and XML files directory
VALID_DIR = '\\val'

# test images for final case
TEST_DIR = '\\test'
# classes: 0 index is reserved for background
CLASSES = [
    'background', 'boneanomaly', 'bonelesion', 'foreignbody', 'fracture', 'metal',
    'periostealreaction', 'pronatorsign', 'axis', 'softtissue', 'text'
]
NUM_CLASSES = 11
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True
# location to save model and plots
OUT_DIR = '/content/outputs'
SAVE_PLOTS_EPOCH = 2  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2  # save model after these many epochs
MODEL_NAME = 'model'


ssl._create_default_https_context = ssl._create_unverified_context


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
# define the training tranforms


def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
# define the validation transforms


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def show_tranformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
            cv2.imshow(sample)


def load_faster_rcnn_model(input_model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        pretrained=True)
    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, len(CLASSES))
    model = nn.DataParallel(model)

    if input_model_path is not None:
        model.load_state_dict(torch.load(
            input_model_path, map_location=torch.device("cpu")))
        model.to(DEVICE)
        model.eval()
    else:
        model.to(DEVICE)
    return model
# function for running training iterations


def train_rcnn(train_data_loader, model, optimiser, train_itr, train_loss_list, train_loss_hist):
    print('Training')

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for _, data in enumerate(prog_bar):
        optimiser.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimiser.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list
# function for running validation iterations


def test_rcnn(test_rcnn_loader, model, test_rcnn_loss_hist, test_rcnn_loss_list, test_itr):
    print('Validating')

    # initialize tqdm progress bar
    prog_bar = tqdm(test_rcnn_loader, total=len(test_rcnn_loader))

    for _, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        test_rcnn_loss_list.append(loss_value)
        test_rcnn_loss_hist.send(loss_value)
        test_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return test_rcnn_loss_list


def train_test_rcnn(model, train_rcnn_loader, test_rcnn_loader):
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimiser = torch.optim.SGD(
        params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    test_rcnn_loss_hist = Averager()
    train_itr = 1
    test_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    test_rcnn_loss_list = []
    # name to save the trained model with

    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        # from utils import show_tranformed_image
        show_tranformed_image(train_rcnn_loader)
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        test_rcnn_loss_hist.reset()
        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, test_ax = plt.subplots()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train_rcnn(train_rcnn_loader, model, optimiser=optimiser, train_itr=train_itr,
                                train_loss_list=train_loss_list, train_loss_hist=train_loss_hist)
        test_loss = test_rcnn(test_rcnn_loader, model, test_rcnn_loss_hist=test_rcnn_loss_hist,
                              test_rcnn_loss_list=test_rcnn_loss_list, test_itr=test_itr)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(
            f"Epoch #{epoch} validation loss: {test_rcnn_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # evaluate(model, test_rcnn_loader, device=DEVICE)
        if (epoch+1) % SAVE_MODEL_EPOCH == 0:  # save model after every n epochs
            torch.save(model.state_dict(),
                       f"{OUT_DIR}/{MODEL_NAME}{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')

        if (epoch+1) % SAVE_PLOTS_EPOCH == 0:  # save loss plots after n epochs
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            test_ax.plot(test_loss, color='red')
            test_ax.set_xlabel('iterations')
            test_ax.set_ylabel('test loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/test_loss_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...')

        if (epoch+1) == NUM_EPOCHS:  # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            test_ax.plot(test_loss, color='red')
            test_ax.set_xlabel('iterations')
            test_ax.set_ylabel('test loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/test_loss_{epoch+1}.png")
            torch.save(model.state_dict(),
                       f"{OUT_DIR}/{MODEL_NAME}{epoch+1}.pth")

        plt.close('all')


def get_iou_for_num_batch_in_loader(model, data_loader, detection_threshold):
    total_count, boneanomaly_count, bonelesion_count, foreignbody_count, fracture_count, metal_count = 0, 0, 0, 0, 0, 0
    periostealreaction_count, pronatorsign_count, axis_count, softtissue_count, text_count = 0, 0, 0, 0, 0
    total_iou, boneanomaly_iou, bonelesion_iou, foreignbody_iou, fracture_iou, metal_iou = 0, 0, 0, 0, 0, 0
    periostealreaction_iou, pronatorsign_iou, axis_iou, softtissue_iou, text_iou = 0, 0, 0, 0, 0
    metric = IntersectionOverUnion(class_metrics=True)
    map_metric = MeanAveragePrecision(class_metrics=True)
    with torch.no_grad():
        for index, item in enumerate(data_loader):

            target = [data_loader.dataset[index][1]]
            val = torch.unsqueeze(data_loader.dataset[index][0], 0)
            res = model(val)
            res = [{k: v.to('cpu') for k, v in t.items()} for t in res]

            boxes = res[0]['boxes'].data.numpy()
            scores = res[0]['scores'].data.numpy()
            labels = res[0]['labels'].data.numpy()
            res[0]['boxes'].data = torch.tensor(
                boxes[scores >= detection_threshold])
            res[0]['scores'].data = torch.tensor(
                scores[scores >= detection_threshold])
            res[0]['labels'].data = torch.tensor(
                labels[scores >= detection_threshold])

            answer = metric(target, res)
            map_metric.update(res, target)
    pprint(map_metric.compute())


def filter_categories(boxes, labels, scores, categories):
    to_keep = []
    for index in range(len(boxes)):
        if labels[index] in categories:
            to_keep.append(index)
    ret_list_boxes = [boxes[i] for i in to_keep]
    ret_list_labels = [labels[i] for i in to_keep]
    ret_list_scores = [scores[i] for i in to_keep]
    return ret_list_boxes, ret_list_labels, ret_list_scores


def run_one_image(model, image_path, detection_threshold, categories):
    image_name = image_path.split('/')[-1]
    image = cv2.imread(image_path)
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(
        orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    if torch.cuda.is_available():
        image = image.cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        # filter out boxes if they're in the categories to be seen
        boxes, labels, scores = filter_categories(
            boxes, labels, scores, categories)

        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i]
                        for i in labels]
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, str(pred_classes[j]) + " "
                        + str(round(scores[j]*100, 2)) + "%",
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, lineType=cv2.LINE_AA)

        path = str(Path(os.getcwd() + "/static/results"))
        try:
            cv2.imwrite(os.path.join(
                path, "modeldetectionimage.jpg"), orig_image)
        except Exception as e:
            print(e)

        return pred_classes, scores


def run_metrics_loader(model, loader, detection_threshold):

    for inputs, labels in loader:
        output = model(inputs)
        output = [{k: v.to('cpu') for k, v in t.items()} for t in output]
        metric = IntersectionOverUnion(class_metrics=True)
        print(metric(output, labels))


def run_tests(model, test_images, detection_threshold):
    path = ""
    for index, img in enumerate(test_images):
        # get the image file name for saving output later on
        _, fe = os.path.splitext(img)
        if fe == ".xml":
            print("xml file")
        elif fe == ".jpg":
            image_name = img.split('/')[-1]
            image = cv2.imread(img)
            orig_image = image.copy()
            # BGR to RGB
            image = cv2.cvtColor(
                orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # make the pixel range between 0 and 1
            # bring color channels to front
            image = np.transpose(image, (2, 0, 1)).astype(float)
            # convert to tensor
            image = torch.tensor(image, dtype=torch.float).cuda()
            # add batch dimension
            image = torch.unsqueeze(image, 0)
            with torch.no_grad():
                outputs = model(image)
            # load all detection to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            # carry further only if there are detected boxes
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                # filter out boxes according to `detection_threshold`
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                # get all the predicited class names
                pred_classes = [CLASSES[i]
                                for i in outputs[0]['labels'].cpu().numpy()]

                # draw the bounding boxes and write the class name on top of it
                for j, box in enumerate(draw_boxes):
                    cv2.rectangle(orig_image,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 0, 255), 2)
                    cv2.putText(orig_image, str(pred_classes[j]) + " "
                                + str(round(scores[j]*100, 2)) + "%",
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                2, lineType=cv2.LINE_AA)

                cv2.imwrite(os.path.join(path, image_name), orig_image)
        print(f"Image {test_images.index(img)} done...")
        print('-'*50)
    print('TEST PREDICTIONS COMPLETE')


def run():
    # prepare the final datasets and data loaders
    train_rcnn_dataset = FracturedDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    valid_rcnn_dataset = FracturedDataset(
        VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    test_rcnn_dataset = FracturedDataset(
        TEST_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    test_rcnn_subset = FracturedDataset("\\test_subset",
                                        RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    train_rcnn_loader = DataLoader(
        train_rcnn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    valid_rcnn_loader = DataLoader(
        valid_rcnn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_rcnn_loader = DataLoader(
        test_rcnn_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_rcnn_subset_loader = DataLoader(test_rcnn_subset, batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=4,
                                         collate_fn=collate_fn)
    TEST_IMG_DIR = "test_images"
    test_images = glob.glob(f"{TEST_IMG_DIR}/*")
    print(f"Test instances: {len(test_images)}")
    # classes: 0 index is reserved for background

    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = 0.75
    torch.cuda.empty_cache()
    model = load_faster_rcnn_model(
        "models\\detection_model.pth")
    run_tests(model, test_images, detection_threshold=detection_threshold)
    # evaluate(model, test_rcnn_loader, DEVICE)
    print("0%")
    get_iou_for_num_batch_in_loader(model, valid_rcnn_loader, 0.0)
    print("25%")
    get_iou_for_num_batch_in_loader(model, valid_rcnn_loader, 0.25)
    print("50%")
    get_iou_for_num_batch_in_loader(model, valid_rcnn_loader, 0.50)
    print("75%")
    get_iou_for_num_batch_in_loader(model, valid_rcnn_loader, 0.75)
    print("90%")
    get_iou_for_num_batch_in_loader(model, valid_rcnn_loader, 0.90)
    print("95%")
    get_iou_for_num_batch_in_loader(model, valid_rcnn_loader, 0.95)
    run_one_image(model, "test_subset\\0001_1297860435_01_WRI-L2_M014.jpg",
                  0.75, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    run_metrics_loader(model, test_rcnn_subset_loader, 0.75)


def main():
    print(DEVICE)
    run()


if __name__ == "__main__":
    main()
