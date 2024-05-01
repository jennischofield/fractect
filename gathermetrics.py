from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, auc, average_precision_score, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve
import torch
from fasterrcnn import load_faster_rcnn_model
from traintest import load_resnext_model
from torchvision.transforms import v2
from adjustimage import AdjustImage
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as nnf

from confidencelevels import get_conf_for_image
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def main():
    model = load_resnext_model(DEVICE, "models\\classification_model.pth")
    path = '\\jpgwmulti'
    transforms = v2.Compose(
        [AdjustImage(), v2.Resize([256, 256]), v2.PILToTensor()])
    xray_dataset = datasets.ImageFolder(path, transform=transforms)
    train_ds_resnext, test_ds_resnext, val_ds_resnext = torch.utils.data.random_split(
        xray_dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
    batch_size_resnext = 32
    train_dataloader_resnext = DataLoader(
        train_ds_resnext, batch_size=batch_size_resnext)
    test_dataloader_resnext = DataLoader(
        test_ds_resnext, batch_size=batch_size_resnext)
    val_dataloader_resnext = DataLoader(
        val_ds_resnext, batch_size=batch_size_resnext)
    classification_metrics(model, val_dataloader_resnext, DEVICE)


def classification_metrics(model, val_dataloader, device):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_dataloader, 0):
            model = model.to(device)
            output = model(inputs.float().to(device))  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)

    # Generate confusion matrix:
    threshold = .85
    indices = []
    for index in range(len(y_pred)):
        if y_pred[index] != y_true[index]:
            print(y_pred[index], y_true[index],
                  val_dataloader.dataset.indices[index])
            indices.append(val_dataloader.dataset.indices[index])
    # print(len(indices))
    print(val_dataloader.dataset)
    print(val_dataloader.dataset.imgs)
    for index in indices:
        print(val_dataloader.dataset.dataset.imgs[index])

    # y_pred_class = y_pred > threshold
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Fractured', 'Not Fractured']
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=labels)
    display.plot()
    plt.show()
    tn, fp, fn, tp = cm.ravel()
    print(
        f"True Negatives:{tn}, False Positives:{fp}, False Negatives:{fn}, True Positives:{tp}")
    print(f"False Positive Rate (Type I Error): {fp / (fp + tn)}")
    print(f"False Negative Rate (Type II Error): {fn / (tp + fn)}")
    print(f"True Negative Rate: {tn / (tn + fp)}")
    print(f"Negative Predictive Value: {tn / (tn + fn)}")
    print(f"False Discovery Rate: {fp / (tp + fp)}")
    print(f"Recall: {tp / (tp + fn)}")
    print(f"Precision: {tp / (tp + fp)}")
    print(f"Accuracy:{(tp + tn) / (tp + fp + fn + tn)}")
    print(f"F1-Score: {f1_score(y_true, y_pred)}")
    print(f"Cohen-Kappa: {cohen_kappa_score(y_true, y_pred)}")
    print(f"MCC:{matthews_corrcoef(y_true, y_pred)}")
    print(f"ROC-AUC:{roc_auc_score(y_true, y_pred)}")
    print(f"Average Precision:{average_precision_score(y_true, y_pred)}")
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                              estimator_name='classifier')
    display.plot()
    plt.show()
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    plt.fill_between(recall, precision)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Test Precision-Recall curve")
    plt.show()
# def detection_metrics(model, val_dataloader,device):


if __name__ == "__main__":
    main()
