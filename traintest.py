from torchvision.transforms import v2
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from adjustimage import AdjustImage
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def load_resnext_model(device, input_model_path):
    ret_model = torch.hub.load(
        'pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    ret_model = nn.DataParallel(ret_model)
    if input_model_path is None:
        ret_model.to(device)
    else:
        ret_model.load_state_dict(torch.load(
            input_model_path, map_location=torch.device("cpu")))
        ret_model.to(device)
    return ret_model

# Both train_resnext and test_resnext adapted from the quickstart section for Pytorch


def train_resnext(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.type(torch.FloatTensor), y.type(torch.FloatTensor)
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.type(torch.LongTensor).to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_resnext(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.type(torch.FloatTensor), y.type(torch.FloatTensor)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,
                                 y.type(torch.LongTensor).to(device)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


def train_test_resnext(model, train_dataloader, test_dataloader, loss_fn, optimiser, scheduler, device):
    epochs = 50
    model.to(device)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_resnext(train_dataloader, model, loss_fn,
                      optimiser, device=device)
        loss = test_resnext(test_dataloader, model, loss_fn, device=device)
        scheduler.step(loss)
    print("Done!")


def generate_confusion_matrix(model, test_dataloader, device):

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in test_dataloader:
        model = model.to(device)

        output = model(inputs.float().to(device))  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ['Fractured', "Not Fractured"]

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    cmdisplay = ConfusionMatrixDisplay(
        confusion_matrix=cf_matrix, display_labels=classes)
    cmdisplay.plot()
    plt.savefig("/model_results/confusion_matrix_resnext")


def save_model(model, path):
    torch.save(model.state_dict(), path)


def main():
    path = '{path to the JPG files }'
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
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = load_resnext_model(
        device, '\\model_with_multi_fracture.pth')

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min',
                                                           factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')
    flag = False

    if flag:
        train_test_resnext(model, train_dataloader=train_dataloader_resnext,
                           test_dataloader=test_dataloader_resnext, loss_fn=loss_fn, optimiser=optimiser, scheduler=scheduler, device=device)


if __name__ == "__main__":
    main()
