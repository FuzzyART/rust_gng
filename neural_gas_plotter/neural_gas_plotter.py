import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import json
import argparse
import os

class ScatterPlot:
    def __init__(self, data_file):
        self.data_file = data_file
        self.plot = plt.figure()
        self.ax = self.plot.add_subplot(111)
        self.points = None
        self.edges = None
        self.edge_positions = None

    def load_data(self):
        if not os.path.exists(self.data_file):
            print(f"Data file {self.data_file} does not exist.")
            return False

        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)

            # Extract neuron positions
            rows = []
            for neuron in data["model"]["neurons"]:
                position = neuron["position"]
                id = neuron['id']
                x = position[0]
                y = position[1]
                rows.append({"id": id, "x": x, "y": y})

            self.points = pd.DataFrame(rows)

            # Extract edges
            row_edge = []
            for edge in data['model']['edges']:
                edge_from = edge['from']
                edge_to = edge['to']
                row_edge.append({"start": edge_from, "to": edge_to})

            self.edges = pd.DataFrame(row_edge)

            # Calculate edge positions
            row = []
            for _, row_edge in self.edges.iterrows():
                try:
                    row_from = self.points.loc[self.points['id'] == row_edge['start']].iloc[0]
                    x_from = float(row_from['x'])
                    y_from = float(row_from['y'])
                    row_to = self.points.loc[self.points['id'] == row_edge['to']].iloc[0]
                    x_to = float(row_to['x'])
                    y_to = float(row_to['y'])

                    row.append({"x_from": x_from, "y_from": y_from, "x_to": x_to, "y_to": y_to})
                except (IndexError, KeyError):
                    continue

            self.edge_positions = pd.DataFrame(row)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def update(self, frame):
        # Reload data every frame
        if not self.load_data():
            return  # Skip if failed to load

        try:
            self.ax.clear()

            # Plot points
            self.ax.scatter(self.points['x'], self.points['y'])

            # Plot connections
            for _, row in self.edges.iterrows():
                try:
                    row_from = self.points.loc[self.points['id'] == row['start']].iloc[0]
                    x_from = float(row_from['x'])
                    y_from = float(row_from['y'])
                    row_to = self.points.loc[self.points['id'] == row['to']].iloc[0]
                    x_to = float(row_to['x'])
                    y_to = float(row_to['y'])

                    self.ax.plot([x_from, x_to], [y_from, y_to], 'k-', alpha=0.6)
                except (IndexError, KeyError):
                    continue

            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_title('Scatter Plot with Connections')

        except Exception as e:
            print(f"Error during update: {e}")

    def run(self):
        # Create animation that updates every 1000ms (1 second)
        ani = animation.FuncAnimation(
            self.plot,
            self.update,
            interval=1000,  # Update every second
            cache_frame_data=False
        )
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process two files.')
    parser.add_argument('data_file', help='The data file to process.')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    args = parser.parse_args()
    data_file = args.data_file

    scatter_plot = ScatterPlot(data_file)
    scatter_plot.run()

if __name__ == "__main__":
    main()

