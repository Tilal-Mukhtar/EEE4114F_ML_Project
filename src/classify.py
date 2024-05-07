import torch
from preprocess import preprocess
from train import ArtificialNeuralNetwork

def main():
    classes = ["circle", "kite", "parallelogram", "rectangle", "rhombus", "square", "triangle"]
    inputSize = 56*56
    outputSize = 8
    hiddenSize = 1000
    model = ArtificialNeuralNetwork(inputSize, hiddenSize, outputSize)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    
    filepath = input("Please enter a filepath:\n")
    while filepath.lower() != "exit":
        try:
            # Read the input image
            image = preprocess(filepath).float().unsqueeze(0)

            # Apply the model to the input
            with torch.no_grad():
                output = model(image)

            # Give the classification of the input image
            classification = torch.argmax(torch.round(output), dim=1).item()
            print("Shape: " + classes[classification])

        except:
            # If the reading the input image throws an exception
            print("Could not load the file.")

        filepath = input("\nPlease enter a filepath:\n")
    print("Exiting...")
    
if __name__ == "__main__":
    main()