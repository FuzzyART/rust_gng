import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import json
import argparse
class ScatterPlot:
    def __init__(self, data_file):
        self.data_file = data_file
        self.plot = plt.figure()
        self.ax = self.plot.add_subplot(111)
        self.data = None
        self.connections = None

        self.points = {}
        self.edges= {}
        self.edge_positions = {}



    def load_data(self):

        filename = self.data_file

        # Load JSON data
        with open(filename, "r") as f:
            data = json.load(f)

        # Extract x and y coordinates
        rows = []
        for neuron in data["model"]["neurons"]:
            position = neuron["position"]
            id = neuron['id']
            x = position[0]
            y = position[1]
            rows.append({"id": id,"x": x, "y": y})

        # Create DataFrame
        self.points = pd.DataFrame(rows)

        row_edge = []
        for edge in data['model']['edges']:
            edge_from = edge['from']
            edge_to = edge['to']
            row_edge.append({"start": edge_from, "to":edge_to})

        self.edges = pd.DataFrame(row_edge)


        #print(self.edges.head(3))
        #print(self.points.head(3))

        row = []
        for rows in self.edges.itertuples():

            row_from = self.points.loc[self.points['id'] == rows.start].iloc[0]
            x_from = float(row_from['x'])
            y_from = float(row_from['y'])
            row_to = self.points.loc[self.points['id'] == rows.to].iloc[0]
            x_to = float(row_to['x'])
            y_to = float(row_to['y'])

            row.append({"x_from": x_from,"y_from": y_from,"x_to": x_to,"y_to": y_to})

        self.edge_positions = pd.DataFrame(row)
        #print("edge pos",self.edge_positions.head(3))


    def update(self):
        try:
            # Read data and connections files
            #self.data = pd.read_csv(self.data_file)
            #self.connections = pd.read_csv(self.connections_file)

            # Clear the previous plot
            self.ax.clear()

            # Plot points
            self.ax.scatter(self.points['x'], self.points['y'])

            # Plot connections
            for rows in self.edges.itertuples():

                row_from = self.points.loc[self.points['id'] == rows.start].iloc[0]
                x_from = float(row_from['x'])
                y_from = float(row_from['y'])
                row_to = self.points.loc[self.points['id'] == rows.to].iloc[0]
                x_to = float(row_to['x'])
                y_to = float(row_to['y'])

                #row.append({"x_from": x_from,"y_from": y_from,"x_to": x_to,"y_to": y_to})
                self.ax.plot([x_from, x_to], [y_from, y_to], 'k-')

                ## Set labels and title for clarity
                #self.ax.set_xlabel('X')
                #self.ax.set_ylabel('Y')
                #self.ax.set_title('Scatter Plot with Connections')

        except Exception as e:
            print(f"Error: {e}")

    


#
    def run(self):
        # Create an animation that updates every 1000ms (1 second)
        ani = animation.FuncAnimation(self.plot, self.update, interval=1000)
        plt.show()

def test():
    filename = '/tmp/output.json'
    scatter_plot = ScatterPlot(filename)
    scatter_plot.load_data()
    scatter_plot.update()
    scatter_plot.run()

if __name__ == "__main__":
    test()
    #main()
