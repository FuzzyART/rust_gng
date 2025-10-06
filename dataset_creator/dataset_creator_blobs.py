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

def create_blobs(filename='dataset.csv',
                 num_samples=100, 
                 num_centers=3, 
                 std_deviation =0.8,
                 rng_seed = 123 
                 ):
    np.random.seed(rng_seed)  # Set random seed for reproducibility



    print("creating blob")
    # Generate and plot blob data
    blob_X, blob_y = generate_blob_data(
        n_samples=num_samples,
        centers=num_centers,
        std_dev=std_deviation
    )
    plot_dataset(blob_X, blob_y, "Blob Classification Dataset")
    #plt.savefig('blobs.png')
    #plt.close()

    x_blob = np.array(blob_X)
    y_blob = np.array(blob_y)

    save_to_csv(x_blob,y_blob,filename=filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Enter the filename")
    parser.add_argument("-s", "--num_samples", help="Enter number of samples" )
    parser.add_argument("-c", "--num_centers", help="Enter number of centers")
    parser.add_argument("-d", "--std_dev", help="Enter value for standard deviation")
    parser.add_argument("-r", "--rng_seed", help="Enter value for the rng seed")
    args = parser.parse_args()

    dataset_filename = args.filename
    n_samples = int(args.num_samples)
    n_centers = int(args.num_centers)
    std_deviation = float(args.std_dev)
    rng = int(args.rng_seed)

    create_blobs(
                 filename       = dataset_filename,
                 num_samples    = n_samples,
                 num_centers    = n_centers,
                 std_deviation  = std_deviation,
                 rng_seed       = rng)



if __name__ == "__main__":
    main()
