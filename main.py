import torch
import itertools
from torchvision import models, transforms
import numpy as np
from scipy.stats import rankdata
import random
import torchvision.utils as vutils
import os

# Load ResNet18 model
from torchvision.datasets import ImageFolder

model = models.resnet18(pretrained=True)

# Set model to evaluation mode
model.eval()

# Define the transform to preprocess the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the image dataset and apply the transform
dataset = ImageFolder('images\\Images\\', transform=transform)
# Select a random subset of the dataset
subset_size = 500
subset_indices = random.sample(range(len(dataset)), subset_size)
subset = torch.utils.data.Subset(dataset, subset_indices)
# Get the class names from the image dataset
class_names = dataset.classes
# Select some base images to mark as target images
target_indices = [10, 15, 27, 30]

# Compute the feature vectors for all images in the dataset using ResNet18
features = []
for image, _ in subset:
    # Add batch dimension to image tensor
    image = image.unsqueeze(0)

    # Pass the image through the ResNet18 model
    with torch.no_grad():
        feature = model(image).squeeze().numpy()

    # Append the feature vector to the list of features
    features.append(feature)

# Convert the list of features to a numpy array
features = np.array(features)

# Normalize the features by rank
rank_features = np.apply_along_axis(rankdata, axis=1, arr=features, method='average') / len(features)

# Construct the hypergraph
n, d = rank_features.shape
similarity_matrix = np.dot(rank_features, rank_features.T) / np.outer(np.linalg.norm(rank_features, axis=1),
                                                                      np.linalg.norm(rank_features, axis=1))
similarity_matrix = np.maximum(similarity_matrix, similarity_matrix.T)

threshold = np.median(similarity_matrix)
adj_matrix = (similarity_matrix >= threshold).astype(float)
np.fill_diagonal(adj_matrix, 0)

# Compute the hyperedge similarities
hyperedge_similarities = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        if adj_matrix[i, j] == 1:
            common_neighbors = np.intersect1d(np.where(adj_matrix[i, :] == 1), np.where(adj_matrix[j, :] == 1))
            if len(common_neighbors) > 0:
                for k in common_neighbors:
                    weight = np.sqrt(similarity_matrix[i, k] * similarity_matrix[j, k])
                    hyperedge_similarities[i, j] += weight
                    hyperedge_similarities[j, i] += weight
# Select all images in the subset
subset_indices = list(range(len(subset)))

# Compute the relevance score for each image
image_scores = []
for image_idx in subset_indices:
    hyperedge_elements = np.where(adj_matrix[image_idx, :] == 1)[0]
    relevance_score = 0
    # Calculates Cartesian product of the lists contained in edge lists
    for i, j in itertools.product(hyperedge_elements, repeat=2):
        if i != j and adj_matrix[i, j] == 1:
            relevance_score += hyperedge_similarities[i, j]
    image_scores.append((image_idx, relevance_score))

# Sort the images by relevance score in descending order
sorted_images = sorted(image_scores, key=lambda x: x[1], reverse=True)

# Print the sorted list of images and their relevance scores
for image in sorted_images:
    print(f"Image index: {image[0]}, Relevance score: {image[1]}")


# Compute the relevance score for each base image
target_scores = []
for target_idx in target_indices:
    hyperedge_elements = np.where(adj_matrix[target_idx, :] == 1)[0]
    relevance_score = 0
    for i, j in itertools.product(hyperedge_elements, repeat=2):
        if i != j and adj_matrix[i, j] == 1:
            relevance_score += hyperedge_similarities[i, j]
    target_scores.append((target_idx, relevance_score))

# Sort the base images by relevance score in descending order
sorted_targets = sorted(target_scores, key=lambda x: x[1], reverse=True)

# Print the sorted list of base images and their relevance scores
for target in sorted_targets:
    print(f"Image index: {target[0]}, Relevance score: {target[1]}")

# Compute accuracy
'''This modification checks if the class of the target image
is the same as the class of the most relevant image, and only increments the correct count if they match.
The final accuracy is computed as the ratio of correct predictions to the total number of target images.'''
correct_count = 0
for target, (idx, score) in zip(target_indices, sorted_targets):
    # Get the class of the target image
    target_class = dataset.classes[target]
    # Get the class of the most relevant image
    relevant_class = dataset.classes[idx]

    if target_class == relevant_class:
        correct_count += 1

accuracy = correct_count / len(target_indices)
print(f"Accuracy: {accuracy}")

# Load the images and their relevance scores
'''This will open a separate window to display each image,
 along with the corresponding relevance score for the base images.'''
target1_idx = sorted_targets[0][0]
target2_idx = sorted_targets[1][0]
target3_idx = sorted_targets[2][0]
target4_idx = sorted_targets[3][0]


best_relevance_idx1 = np.argmax(hyperedge_similarities[target1_idx])
best_relevance_idx2 = np.argmax(hyperedge_similarities[target2_idx])
best_relevance_idx3 = np.argmax(hyperedge_similarities[target3_idx])
best_relevance_idx4 = np.argmax(hyperedge_similarities[target4_idx])

# Load the base images
base_image1, _ = subset[target1_idx]
base_image2, _ = subset[target2_idx]
base_image3, _ = subset[target3_idx]
base_image4, _ = subset[target4_idx]

# Load the best relevance score image
best_relevance_image1, _ = subset[best_relevance_idx1]
best_relevance_image2, _ = subset[best_relevance_idx2]
best_relevance_image3, _ = subset[best_relevance_idx3]
best_relevance_image4, _ = subset[best_relevance_idx4]

# Display the images
images = [base_image1, base_image2,base_image3, base_image4, best_relevance_image1,
          best_relevance_image2, best_relevance_image3, best_relevance_image4]
grid = vutils.make_grid(images, nrow=4, padding=5, normalize=True)
vutils.save_image(grid, 'images.png')

# Open the saved image file
os.startfile('images.png')
