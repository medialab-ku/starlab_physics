import matplotlib
import matplotlib.pyplot as plt
import json
from datetime import datetime

matplotlib.use('Agg')

# Sample data
# data_1 = {
#     "name": 1234,
#     "label": "PCG",
#     "conditions": {
#         "precond_type" : "PCG",
#         "condition_1" : "Condition 1",
#         "condition_2" : "Condition 2"
#     },
#     "data": {
#         1:20,
#         2:40,
#         3:50,
#         4:30,
#         5:35,
#     }
# }
# data_2 = {
#     "name": 5678,
#     "conditions": {
#         "precond_type" : "Our solver",
#         "condition_1" : "Condition 1",
#         "condition_2" : "Condition 2"
#     },
#     "data": {
#         1:10,
#         2:20,
#         3:40,
#         4:20,
#         5:15,
#     }
# }

class make_plot:
    def __init__(self, output_path, x_name: str, y_name: str, graph_name = "Graph"):
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        self.output_path = output_path
        self.x_name = x_name
        self.y_name = y_name
        self.graph_name = graph_name

        self.graph_type = "line"
        if self.x_name == "frame" and self.y_name == "energy":
            self.graph_type = "line"
        elif self.x_name == "frame" and self.y_name == "iteration":
            self.graph_type = "line"
        elif self.x_name == "iteration" and self.y_name == "energy":
            self.graph_type = "line"

        self.aggregated_data = {}

        # self.collect_data(data_1["conditions"], data_1["data"])
        # self.collect_data(data_2["conditions"], data_2["data"])

        # plot = self.make_graph()
        # if plot is None:
        #     print("The graph is not correctly created!")
        #     print("You should not change the end frame number during collecting data...")
        #     return
        # self.export_result(plot)

    def collect_data(self, data: dict):
        name = data["name"]
        if name in self.aggregated_data:
            raise NameError(f"Name Error : '{name}' already exists in the aggregated data.")
        else:
            self.aggregated_data[data["name"]] = data
            print("The result data is collected successfully!")

    def make_graph(self):
        x = None
        y = []
        labels = []

        for name, cond_and_data in self.aggregated_data.items():
            labels.append(cond_and_data["label"])

            if x == None:
                x = list(cond_and_data["data"].keys())
            elif x == list(cond_and_data["data"].keys()):
                pass
            else:
                print("The x-axis data is not equal to other data!")
                return None

            y.append(list(cond_and_data["data"].values()))

        for i, label in enumerate(labels):
            color = self.color_palette[i % len(self.color_palette)]
            if self.graph_type == "line":
                plt.plot(x, y[i], color=color, label=label)
            elif self.graph_type == "bar":
                plt.bar(x, y[i], color=color, label=label)

        min_value = min(min(y_graph) for y_graph in y)
        max_value = max(max(y_graph) for y_graph in y)

        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.ylim(min_value, max_value)
        plt.title(self.graph_name) # will be revised soon
        plt.legend()

        plot = plt.gcf() # make the plot object
        return plot

    def export_result(self, plot):
        current_time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        json_file_path = self.output_path + current_time_str + ".json"
        plot_file_path = self.output_path + current_time_str + ".png"

        # export the dictionary to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(self.aggregated_data, json_file, indent=4)

        # export the plot image
        plot.savefig(plot_file_path)
        plt.close(plot)

        self.aggregated_data = {}

# graph = make_plot("../../results/", "iter", "energy")