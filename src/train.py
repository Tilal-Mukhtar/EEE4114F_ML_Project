"""
File: classifier.py
Author: Tilal Zaheer Mukhtar
Date: 14/04/2024
Description: The implementation of a feedforward ANN that can classify handwritten digits from the MNIST10 dataset
"""
import os
import torch
import torch.nn as nn


# Define the feedforward ANN model
class ArtificialNeuralNetwork(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super().__init__(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )


# Calculate the loss and accuracy of the ANN model for the given dataset
def get_model_performance(model, loss_function, dataset):
    # Apply the model to the dataset
    images, labels = dataset[:]
    outputs = model(images)

    # Calculate the loss
    loss = loss_function(outputs, labels)

    # Calculate the accuracy
    predicted = torch.argmax(torch.round(outputs), dim=1)
    total = labels.size(0)
    correct = (predicted == labels).sum()
    accuracy = 100 * correct / total

    return loss, accuracy


# Main function
def main():
    # Check for available cuda device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the input and output sizes of the feedforward ANN model
    inputSize = 56*56
    outputSize = 8

    # Define the hyperparameters of the feedforward ANN model
    hiddenSize = 1000
    epochs = 100
    batchSize = 16
    learningRate = 0.001
    dropout = 0.5
    earlyStop = 10

    # Define the classification model, loss function, and optimization algorithm
    model = ArtificialNeuralNetwork(
        inputSize, hiddenSize, outputSize, dropout).to(device)
    lossFunction = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

    # Create datasets
    print("Loading the dataset...")
    directory = os.path.join(".", "dataset", "processed")
    trainDataset = torch.load(os.path.join(directory, "train.pt"))
    valDataset = torch.load(os.path.join(directory, "val.pt"))
    testDataset = torch.load(os.path.join(directory, "test.pt"))
    trainDataset = torch.utils.data.TensorDataset(
        trainDataset[0].to(device).float(), trainDataset[1].to(device))
    valDataset = torch.utils.data.TensorDataset(
        valDataset[0].to(device).float(), valDataset[1].to(device))
    testDataset = torch.utils.data.TensorDataset(
        testDataset[0].to(device).float(), testDataset[1].to(device))
    print("Training Dataset Size:   {}".format(len(trainDataset)))
    print("Validation Dataset Size: {}".format(len(valDataset)))
    print("Testing Dataset Size:    {}".format(len(testDataset)))
    print()

    # Create dataloaders
    trainLoader = torch.utils.data.DataLoader(
        dataset=trainDataset, batch_size=batchSize, shuffle=True)

    # Train the ANN model
    print("Training the model...")
    steps = len(trainLoader)
    stopCounter = 0
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(trainLoader):
            # Forward pass
            outputs = model(images)
            loss = lossFunction(outputs, labels)
            optimizer.zero_grad()

            # Backpropogate the loss
            loss.backward()

            # Update model weights
            optimizer.step()

            # Display current training loss
            if (step+1) % 50 == 0:
                print("Epoch: [{}/{}], Step: [{:3d}/{}], Training Batch Loss: {:.4f}".format(
                    epoch+1, epochs, step+1, steps, loss.item()))

        # Store previous validation loss
        if epoch > 0:
            prev_val_loss = val_loss

        # Compare the loss and accuracy for the training and validation datasets
        with torch.no_grad():
            model.eval()  # Turn off the dropout for validation
            train_loss, train_accuracy = get_model_performance(
                model, lossFunction, trainDataset)
            val_loss, val_accuracy = get_model_performance(
                model, lossFunction, valDataset)
            model.train()

        # Print the training and validation loss and accuracy
        print()
        print("Epoch:               [{}/{}]".format(epoch+1, epochs))
        print("Training Loss:       {:.4f}".format(train_loss.item()))
        print("Validation Loss:     {:.4f}".format(val_loss.item()))
        print("Training Accuracy:   {:.2f} %".format(train_accuracy))
        print("Validation Accuracy: {:.2f} %".format(val_accuracy))
        print()

        # Increment early stopping counter if the validation loss has not decreased
        if epoch > 0 and val_loss.item() >= prev_val_loss.item():
            stopCounter += 1
        # Reset the early stopping counter if the validation loss has decreased
        else:
            stopCounter = 0
        # Check if the validation loss has not decreased for specified number of consecutive epochs
        if stopCounter == earlyStop:
            break

    # Test the feedforward ANN model
    print("Testing the model...")
    model.eval()
    with torch.no_grad():
        test_loss, test_accuracy = get_model_performance(
            model, lossFunction, testDataset)
    print("Testing Loss:     {:.4f}".format(test_loss))
    print("Testing Accuracy: {:.2f} %".format(test_accuracy))
    print()

    # Save model
    torch.save(model.state_dict(), "model.pt")
    print("Done!")


if __name__ == "__main__":
    main()
