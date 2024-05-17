import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from train import CNN, evaluate
import matplotlib.pyplot as plt


def main():
    # Load the CNN image classification model
    classes = ["circle",  "hexagon", "lightning", "square", "star", "triangle"]
    num_classes = 6
    criterion = nn.CrossEntropyLoss()
    model = CNN(num_classes, dropout_rate=0)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    # Load the QuickDraw test dataset
    print("Loading the QuickDraw dataset...")
    directory = os.path.join(".", "dataset", "processed")
    test_dataset = torch.load(os.path.join(directory, "test.pt"))
    test_dataset = TensorDataset(
        test_dataset[0].float(), test_dataset[1])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Testing Dataset Size:    {}".format(len(test_dataset)))
    print()

    # Evaluate the model on the QuickDraw test dataset
    print("Testing the model...")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print("Testing Loss:     {:.4f}".format(test_loss))
    print("Testing Accuracy: {:.2f} %".format(test_accuracy))
    print()

    # Plot the confusion matrix of the results for the QuickDraw test set
    images, labels = test_dataset[:]
    with torch.no_grad():
        predicted = torch.argmax(torch.round(model(images.unsqueeze(1))), dim=1)
    conf_matrix = confusion_matrix(labels.cpu(), predicted.cpu())
    ConfusionMatrixDisplay(conf_matrix, display_labels=classes).plot()
    plt.show()

    # Load the self-created test dataset
    print("Loading the self-created test dataset...")
    directory = os.path.join(".", "custom_dataset", "processed")
    test_dataset = torch.load(os.path.join(directory, "test.pt"))
    test_dataset = TensorDataset(
        test_dataset[0].float(), test_dataset[1])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print("Testing Dataset Size:    {}".format(len(test_dataset)))
    print()

    # Evaluate the model on the self-created test dataset
    print("Testing the model...")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print("Testing Loss:     {:.4f}".format(test_loss))
    print("Testing Accuracy: {:.2f} %".format(test_accuracy))
    print()

    # Plot the confusion matrix of the results for the self-created test set
    images, labels = test_dataset[:]
    with torch.no_grad():
        predicted = torch.argmax(torch.round(model(images.unsqueeze(1))), dim=1)
    conf_matrix = confusion_matrix(labels.cpu(), predicted.cpu())
    ConfusionMatrixDisplay(conf_matrix, display_labels=classes).plot()
    plt.show()


if __name__ == "__main__":
    main()
