import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


# Process an input image into the required format
def process_image(filepath, width=28, height=28):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = torch.tensor(image, dtype=torch.bool)
    return image


# Process the QuickDraw dataset into a format that can be loaded into a PyTorch dataset
def process_dataset(directory=os.path.join(".", "dataset")):
    rawDirectory = os.path.join(directory, "raw")
    processedDirectory = os.path.join(directory, "processed")
    files = os.listdir(rawDirectory)
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for i in range(len(files)):
        total_size = 10000
        train_size = 8000
        val_size = 0
        test_size = 2000

        data = np.load(os.path.join(rawDirectory, files[i]))
        data = data[np.random.choice(
            data.shape[0], total_size, replace=False)].reshape(-1, 28, 28)
        kernel = np.ones((2, 2), np.uint8)
        data = cv2.dilate(data, kernel, iterations=1)
        data = 1 * (data > 127)

        # Uncomment to view the images
        # plt.imshow(data[0])
        # plt.show()

        indices = np.random.randint(data.shape[0], size=total_size)

        train_images.extend(data[indices[0:train_size]])
        train_labels.extend(i * np.ones(train_size))
        test_images.extend(
            data[indices[train_size+val_size:train_size+val_size+test_size]])
        test_labels.extend(i * np.ones(test_size))

    train_images = torch.tensor(np.array(train_images), dtype=torch.bool)
    train_labels = torch.tensor(train_labels, dtype=torch.uint8)
    test_images = torch.tensor(np.array(test_images), dtype=torch.bool)
    test_labels = torch.tensor(test_labels, dtype=torch.uint8)
    torch.save((train_images, train_labels),
               os.path.join(processedDirectory, "train.pt"))
    torch.save((test_images, test_labels),
               os.path.join(processedDirectory, "test.pt"))


# Process the raw self-created dataset into a format that can be loaded into a PyTorch dataset
def process_custom_dataset(directory=os.path.join(".", "custom_dataset")):
    rawDirectory = os.path.join(directory, "raw")
    processedDirectory = os.path.join(".", "custom_dataset", "processed")
    for folder in os.listdir(rawDirectory):
        subdirectory = os.path.join(rawDirectory, folder)
        classes = os.listdir(subdirectory)
        images = []
        labels = []
        for i in range(len(classes)):
            subsubdirectory = os.path.join(subdirectory, classes[i])
            files = os.listdir(subsubdirectory)
            for file in files:
                images.append(process_image(
                    os.path.join(subsubdirectory, file)))
                labels.append(i)

                # Uncomment to view the images
                # plt.imshow(images[-1])
                # plt.show()

        images = torch.tensor(np.array(images))
        labels = torch.tensor(labels)
        torch.save((images, labels), os.path.join(
            processedDirectory, folder + ".pt"))


def main():
    process_dataset()
    process_custom_dataset()


if __name__ == "__main__":
    main()
