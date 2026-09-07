import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import argparse

class ScatterPlot:
    def __init__(self, data_file):
        self.data_file = data_file
        self.plot = plt.figure()
        self.ax = self.plot.add_subplot(111)
        self.data = None

    def update(self, frame):
        try:
            # Read data file each time (simulate live updates)
            self.data = pd.read_csv(self.data_file)

            # Clear the previous plot
            self.ax.clear()

            # Plot points
            self.ax.scatter(self.data['x'], self.data['y'])
            self.ax.set_title("Live Scatter Plot")

        except Exception as e:
            print(f"Error: {e}")

    def run(self):
        # Create an animation that updates every 1000ms (1 second)
        ani = animation.FuncAnimation(
            self.plot,
            self.update,
            interval=1000,
            cache_frame_data=False
        )
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process two files.')
    parser.add_argument('-f','--data_file', help='The data file to process.')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')

    args = parser.parse_args()
    data_file = args.data_file

    scatter_plot = ScatterPlot(data_file)
    scatter_plot.run()

if __name__ == "__main__":
    main()

