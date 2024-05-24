import os
from torchvision.datasets import CIFAR10

# Directory where the CIFAR-10 data will be saved
base_dir = 'cifar10'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Create the base directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Define a function to save images to the correct directory
def save_images(dataset, split):
    # Define the split directory
    split_dir = train_dir if split == 'train' else test_dir
    
    # Define the mapping of class index to class name
    classes = dataset.classes
    
    for class_index, class_name in enumerate(classes):
        # Create directory for each class
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    for idx, (image, label) in enumerate(dataset):
        class_name = classes[label]
        class_dir = os.path.join(split_dir, class_name)
        image_path = os.path.join(class_dir, f'{idx}.png')
        
        # Save the image
        image.save(image_path)

# Download and save CIFAR-10 train data
train_dataset = CIFAR10(root=base_dir, train=True, download=True)
save_images(train_dataset, 'train')

# Download and save CIFAR-10 test data
test_dataset = CIFAR10(root=base_dir, train=False, download=True)
save_images(test_dataset, 'test')

print("CIFAR-10 dataset downloaded and images saved in the specified directory structure.")
