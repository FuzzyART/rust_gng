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

def create_dataset(filename='dataset.csv',num_samples=100,noise=0.05,factor=0.7,rng=123):
    np.random.seed(123)  # Set random seed for reproducibility

    print("creating circle")

    # Generate and plot circle data
    circle_X, circle_y = generate_circle_data(
        n_samples=num_samples,
        noise=noise,
        factor=factor
    )
    plot_dataset(circle_X, circle_y, "Circle Classification Dataset")

    x_circle = np.array(circle_X)
    y_circle = np.array(circle_y)
  
    save_to_csv(x_circle,y_circle,filename=filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Enter the filename")
    parser.add_argument("-s", "--num_samples", help="Enter number of samples")
    parser.add_argument("-n", "--noise", help="Enter noise level" )
    parser.add_argument("-a", "--factor", help="Enter factor" )
    parser.add_argument("-r", "--rng_seed", help="Enter the rng seed" )
    args = parser.parse_args()

    data_filename = args.filename
    n_samples = int(args.num_samples)
    noise     = float(args.noise)
    factor    = float(args.factor)
    rng       = int(args.rng_seed)
    create_dataset(filename     = data_filename,
                   num_samples  = n_samples,
                   noise        = noise,
                   factor       = factor,
                   rng          = rng
                   )


if __name__ == "__main__":
    main()
