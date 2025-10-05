import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
import csv
import argparse

# Save the dataset to a CSV file
def save_to_csv(X, y, filename='test_output.csv'):
    """Save the dataset to a CSV file"""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file,delimiter=',')
        writer.writerow(['x', 'y'])
        for i in range(len(X)):
            writer.writerow(X[i])

def generate_blob_data(n_samples=300, centers=2, std_dev=1.0, random_state=None):
    """
    Generate blob-like data for classification
    
    Parameters:
        n_samples: Number of data points to generate
        centers: Number of clusters/blobs
        std_dev: Standard deviation of each blob
        random_state: For reproducibility
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=std_dev,
        random_state=random_state
    )
    return X, y

def generate_circle_data(n_samples=300, noise=0.1, factor=0.8, random_state=None):
    """
    Generate circle-like data for classification
    
    Parameters:
        n_samples: Number of data points to generate
        noise: Amount of noise to add
        factor: Size ratio between inner and outer circles
        random_state: For reproducibility
    """
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state
    )
    return X, y

def plot_dataset(X, y, title="Classification Dataset"):
    """Plot the generated dataset"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.colorbar(label='Class')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

def create_dataset(filename='dataset.csv',type='none',num_samples=100):
    np.random.seed(42)  # Set random seed for reproducibility

    if type == 'circle':
        print("creating circle")

        # Generate and plot circle data
        circle_X, circle_y = generate_circle_data(
            n_samples=num_samples,
            noise=0.05,
            factor=0.7
        )
        plot_dataset(circle_X, circle_y, "Circle Classification Dataset")
        #plt.savefig('circles.png')
        #plt.close()

        x_circle = np.array(circle_X)
        y_circle = np.array(circle_y)
  
        save_to_csv(x_circle,y_circle,filename=filename)


    elif type == 'blob':
        print("creating blob")
        # Generate and plot blob data
        blob_X, blob_y = generate_blob_data(
            n_samples=400,
            centers=3,
            std_dev=0.8
        )
        plot_dataset(blob_X, blob_y, "Blob Classification Dataset")
        #plt.savefig('blobs.png')
        #plt.close()

        x_blob = np.array(blob_X)
        y_blob = np.array(blob_y)

        save_to_csv(x_blob,y_blob,filename=filename)
    else:
        print("No valid dataset type provided. Please use 'blob' or 'circle'.")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Enter the filename")
    parser.add_argument("-t", "--type", help="Enter the type of dataset to create (blob/circle)", choices=['blob', 'circle'], default='circle')
    parser.add_argument("-s", "--num_samples", help="Enter number of samples" )
    args = parser.parse_args()

    circle_filename = args.filename
    dataset_type = args.type
    n_samples = int(args.num_samples)
    create_dataset(filename=circle_filename, type=dataset_type,num_samples=n_samples)


if __name__ == "__main__":
    main()
