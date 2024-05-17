import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset


# Define the CNN image classification model
class CNN(nn.Sequential):
    def __init__(self, num_classes, dropout_rate=0):
        super().__init__(
            nn.Dropout(dropout_rate),
            nn.Conv2d(1, 24, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )


# Evaluate the loss and accuracy of the model across the given dataset using the given loss function
def evaluate(model, data_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

    # Return mean loss, accuracy
    return running_loss / len(data_loader), 100 * correct / total


# Main function
def main():
    # Check for available cuda device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the classification classes
    num_classes = 6
    classes = ["circle",  "hexagon", "lightning", "square", "star", "triangle"]

    # Define the hyperparameters for training the CNN model
    num_folds = 4
    max_epochs = 20
    batch_size = 64
    learning_rate = 0.001
    dropout_rate = 0.2

    # Create datasets
    print("Loading the dataset...")
    directory = os.path.join(".", "dataset", "processed")

    train_dataset = torch.load(os.path.join(directory, "train.pt"))
    test_dataset = torch.load(os.path.join(directory, "test.pt"))

    train_dataset = TensorDataset(
        train_dataset[0].to(device).float(), train_dataset[1].to(device))
    test_dataset = TensorDataset(
        test_dataset[0].to(device).float(), test_dataset[1].to(device))

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    print("Training Dataset Size:   {}".format(len(train_dataset)))
    print("Testing Dataset Size:    {}".format(len(test_dataset)))
    print()

    # Initialize the k-fold cross validation
    k_fold = KFold(n_splits=num_folds, shuffle=True)

    # Store the average training and validation history accross all folds
    k_fold_history = {'train_loss': [], 'val_loss': [],
                      'train_accuracy': [], 'val_accuracy': []}

    # Loop through each fold
    print("Training the model...")
    for fold, (train_indices, val_indices) in enumerate(k_fold.split(train_dataset)):
        # Define the data loaders for the current fold
        fold_train_dataset = Subset(
            train_dataset, train_indices)
        fold_val_dataset = Subset(train_dataset, val_indices)
        train_loader = DataLoader(
            dataset=fold_train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=fold_val_dataset,
                                batch_size=batch_size, shuffle=False)

        # Define the classification model, loss function, and optimization algorithm
        model = CNN(num_classes, dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Store the training and validation history for the current fold
        history = {'train_loss': [], 'val_loss': [],
                   'train_accuracy': [], 'val_accuracy': []}

        # Train the model on the current fold
        steps = len(train_loader)
        for epoch in range(max_epochs):
            running_loss = 0.0
            for step, (images, labels) in enumerate(train_loader):
                model.train()

                # Forward pass
                outputs = model(images.unsqueeze(1))
                loss = criterion(outputs, labels)
                optimizer.zero_grad()

                # Backpropogate the loss
                loss.backward()

                # Update model weights
                optimizer.step()

                # Display the results at every 100 mini-batches
                running_loss += loss.item()
                if (step % 100) == 99:
                    # Get the training loss over the current mini-batch
                    train_loss = running_loss / 100
                    history['train_loss'].append(train_loss)
                    running_loss = 0.0

                    # Get the training accuracy over the current mini-batch
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).sum().item()
                    train_accuracy = 100 * correct / labels.size(0)
                    history['train_accuracy'].append(train_accuracy)

                    # Evaluate the CNN model on the validation dataset
                    val_loss, val_accuracy = evaluate(
                        model, val_loader, criterion)
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_accuracy)

                    # Print the training and validation loss and accuracy
                    print(
                        "Fold:                [{}/{}]".format(fold+1, num_folds))
                    print(
                        "Epoch:               [{}/{}]".format(epoch+1, max_epochs))
                    print("Mini-Batch:          [{}/{}]".format(step+1, steps))
                    print("Training Loss:       {:.4f}".format(train_loss))
                    print("Validation Loss:     {:.4f}".format(val_loss))
                    print("Training Accuracy:   {:.2f} %".format(
                        train_accuracy))
                    print("Validation Accuracy: {:.2f} %".format(val_accuracy))
                    print()

        # Update the training and validation history over all folds
        if (len(k_fold_history["train_loss"]) == 0):
            k_fold_history["train_loss"] = history["train_loss"]
            k_fold_history["val_loss"] = history["val_loss"]
            k_fold_history["train_accuracy"] = history["train_accuracy"]
            k_fold_history["val_accuracy"] = history["val_accuracy"]
        else:
            k_fold_history["train_loss"] = np.sum(
                [history["train_loss"], k_fold_history["train_loss"]], axis=0)
            k_fold_history["val_loss"] = np.sum(
                [history["val_loss"], k_fold_history["val_loss"]], axis=0)
            k_fold_history["train_accuracy"] = np.sum(
                [history["train_accuracy"], k_fold_history["train_accuracy"]], axis=0)
            k_fold_history["val_accuracy"] = np.sum(
                [history["val_accuracy"], k_fold_history["val_accuracy"]], axis=0)

    # Average the training and validation history over all folds
    k_fold_history["train_loss"] = np.array(
        k_fold_history["train_loss"])/num_folds
    k_fold_history["val_loss"] = np.array(k_fold_history["val_loss"])/num_folds
    k_fold_history["train_accuracy"] = np.array(
        k_fold_history["train_accuracy"])/num_folds
    k_fold_history["val_accuracy"] = np.array(
        k_fold_history["val_accuracy"])/num_folds

    # Print the final training and validation loss and accuracy values
    print("Final Training Loss:       {:.4f}".format(
        k_fold_history["train_loss"][-1]))
    print("Final Validation Loss:     {:.4f}".format(
        k_fold_history["val_loss"][-1]))
    print("Final Training Accuracy:   {:.2f} %".format(
        k_fold_history["train_accuracy"][-1]))
    print("Final Validation Accuracy: {:.2f} %".format(
        k_fold_history["val_accuracy"][-1]))
    print()

    # Plot the average training and validation losses over all folds
    plt.plot(k_fold_history['train_loss'], label='Training Loss')
    plt.plot(k_fold_history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel("Logging Iterations")
    plt.ylabel("Cross-Entropy Loss")
    plt.show()

    # Plot the average training and validation accuracies over all folds
    plt.plot(k_fold_history['train_accuracy'], label='Training Accuracy')
    plt.plot(k_fold_history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.xlabel("Logging Iterations")
    plt.ylabel("Accuracy (%)")
    plt.show()

    # Evaluate the CNN model on the test set
    print("Testing the model...")
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print("Testing Loss:     {:.4f}".format(test_loss))
    print("Testing Accuracy: {:.2f} %".format(test_accuracy))
    print()

    # Plot the confusion matrix of the results for the test set
    images, labels = test_dataset[:]
    predicted = torch.argmax(torch.round(model(images.unsqueeze(1))), dim=1)
    conf_matrix = confusion_matrix(labels.cpu(), predicted.cpu())
    ConfusionMatrixDisplay(conf_matrix, display_labels=classes).plot()
    plt.show()

    # Save the model parameters
    torch.save(model.state_dict(), "model.pt")
    print("Done!")


if __name__ == "__main__":
    main()
