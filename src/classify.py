from preprocess import process_image
from train import CNN
import torch
import torch.nn as nn


def main():
    # Load the CNN image classification model
    classes = ["circle",  "hexagon", "lightning", "square", "star", "triangle"]
    num_classes = 6
    model = CNN(num_classes, dropout_rate=0)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()

    filepath = input("Please enter a filepath:\n")
    while filepath.lower() != "exit":
        try:
            # Read the input image
            image = process_image(filepath).float().unsqueeze(0).unsqueeze(0)

            # Apply the model to the input
            with torch.no_grad():
                predicted = model(image)

            # Give the classification of the input image
            classification = torch.argmax(torch.round(predicted), dim=1).item()
            print("Shape: " + classes[classification])

        except Exception as e:
            # If the reading the input image throws an exception
            print("Could not load the file.")
            print(e)

        filepath = input("\nPlease enter a filepath:\n")
    print("Exiting...")


if __name__ == "__main__":
    main()
