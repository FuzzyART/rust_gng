import matplotlib.pyplot as plt

def display_gng():
    points = None
    edges = None
    edge_positions = None
    rows = []
    for neuron in data["model"]["neurons"]:
        position = neuron["position"]
        id = neuron['id']
        x = position[0]
        y = position[1]
        rows.append({"id": id, "x": x, "y": y})

    points = pd.DataFrame(rows)

    # Extract edges
    row_edge = []
    for edge in data['model']['edges']:
        edge_from = edge['from']
        edge_to = edge['to']
        row_edge.append({"start": edge_from, "to": edge_to})

    edges = pd.DataFrame(row_edge)

    # Calculate edge positions
    row = []
    for _, row_edge in edges.iterrows():
        try:
            row_from = points.loc[points['id'] == row_edge['start']].iloc[0]
            x_from = float(row_from['x'])
            y_from = float(row_from['y'])
            row_to = points.loc[points['id'] == row_edge['to']].iloc[0]
            x_to = float(row_to['x'])
            y_to = float(row_to['y'])

            row.append({"x_from": x_from, "y_from": y_from, "x_to": x_to, "y_to": y_to})
        except (IndexError, KeyError):
            continue

    edge_positions = pd.DataFrame(row)
    import matplotlib.pyplot as plt
    plot = plt.figure()
    ax = plot.add_subplot(111)

    ax.clear()

    # Plot points
    ax.scatter(points['x'], points['y'])

    # Plot connections
    for _, row in edges.iterrows():
        row_from = points.loc[points['id'] == row['start']].iloc[0]
        x_from = float(row_from['x'])
        y_from = float(row_from['y'])
        row_to = points.loc[points['id'] == row['to']].iloc[0]
        x_to = float(row_to['x'])
        y_to = float(row_to['y'])

        ax.plot([x_from, x_to], [y_from, y_to], 'k-', alpha=0.6)
   
 
def display_gng_classified():
    import matplotlib.pyplot as plt
    plot = plt.figure()
    ax = plot.add_subplot(111)

    ax.clear()



    # Plot points
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    my_norm = mpl.colors.Normalize(vmin=points['class'].min(), 
                                vmax=points['class'].max())
    my_map = ListedColormap(["red","blue"])
    #ax.scatter(points['x'], points['y'],c=points['class'],cmap='hsv',norm=my_norm)
    ax.scatter(points['x'], points['y'],c=points['class'],cmap=my_map,norm=my_norm)

    # Plot connections
    for _, row in edges.iterrows():
        row_from = points.loc[points['id'] == row['start']].iloc[0]
        x_from = float(row_from['x'])
        y_from = float(row_from['y'])
        row_to = points.loc[points['id'] == row['to']].iloc[0]
        x_to = float(row_to['x'])
        y_to = float(row_to['y'])

        ax.plot([x_from, x_to], [y_from, y_to], 'k-', alpha=0.3)
    plt.show()

    
def display_gng_classified_heatmap():
    import matplotlib.pyplot as plt
    plot = plt.figure()
    ax = plot.add_subplot(111)

    ax.clear()



    # Plot points
    ax.scatter(points['x'], points['y'],c=points['class'],cmap='hsv')

    # Plot connections
    for _, row in edges.iterrows():
        row_from = points.loc[points['id'] == row['start']].iloc[0]
        x_from = float(row_from['x'])
        y_from = float(row_from['y'])
        row_to = points.loc[points['id'] == row['to']].iloc[0]
        x_to = float(row_to['x'])
        y_to = float(row_to['y'])

        ax.plot([x_from, x_to], [y_from, y_to], 'k-', alpha=0.3)

        
def display_input_set(in_set):
    plot = plt.figure()
    plot = plt.scatter(in_set[:,0],in_set[:,1])
    plot.show()