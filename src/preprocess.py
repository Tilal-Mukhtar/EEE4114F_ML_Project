import numpy as np
import os
import torch
import torchvision
from skimage.filters import threshold_sauvola

def preprocess(filepath, width=56, height=56, windowSize=15):
    image = torchvision.io.read_image(filepath)
    image = torchvision.transforms.functional.rgb_to_grayscale(image)
    image = torch.nn.functional.interpolate(image.unsqueeze(
        0), size=(width, height), mode='bicubic').squeeze()
    image = 1 * \
        (image < torch.tensor(threshold_sauvola(np.array(image), windowSize)))
    image = image.reshape(-1, width*height).squeeze()
    return image


def main():
    rawDirectory = os.path.join(".", "dataset", "raw")
    processedDirectory = os.path.join(".", "dataset", "processed")
    for folder in os.listdir(rawDirectory):
        subdirectory = os.path.join(rawDirectory, folder)
        classes = os.listdir(subdirectory)
        images = []
        labels = []
        for i in range(len(classes)):
            subsubdirectory = os.path.join(subdirectory, classes[i])
            files = os.listdir(subsubdirectory)
            for file in files:
                images.append(preprocess(os.path.join(subsubdirectory, file)))
                labels.append(i)
        images = torch.tensor(np.array(images))
        labels = torch.tensor(labels)
        torch.save((images, labels), os.path.join(
            processedDirectory, folder + ".pt"))


if __name__ == "__main__":
    main()
