import os
import csv


def count_images_in_directory(directory):
    num_images = 0
    for _, _, files in os.walk(directory):
        for file_name in files:
            num_images += 1
    return num_images

# To make all images in directory 3347 
def remove_random_images(directory, num_images_to_remove):
    for subdir, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(subdir, file_name)
            os.remove(file_path)
            print(f"Removed: {file_path}")
            num_images_to_remove -= 1
            if num_images_to_remove <= 0:
                return


def create_csv(root_dirs, output_file):
    class_one_hot = {"berry": 0, "bird": 1, "dog": 2, "flower": 3, "other": 4}
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['path', 'noisy_labels_0']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for root_dir in root_dirs:
            for dir_path, _, files in os.walk(root_dir):
                class_name = os.path.basename(dir_path)
                class_label = class_one_hot.get(class_name, -1)
                if class_label != -1:
                    for file in files:
                        file_path = os.path.join(dir_path, file)
                        # Replace the folder name with its corresponding class number
                        file_path = file_path.replace(class_name, str(class_label))
                        writer.writerow({'path': file_path, 'noisy_labels_0': class_label})

if __name__ == "__main__":
    create_csv(["train", "test"], "data.csv")
