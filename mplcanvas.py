
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import random

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent = None):
        super().__init__(Figure())
        self.setParent(parent)
        self.figure.subplots_adjust(bottom=0.2)
        self.axes = self.figure.subplots()
        self.axes.set_xlabel('number of iterations')
        self.axes.set_ylabel('train loss')

    def draw_loss(self, train_counter, train_losses, legend_name):
        self.axes.plot(train_counter, train_losses, label=legend_name)
        self.axes.legend()
        self.draw()

    def clear_plot(self):
        self.axes.clear()
        self.draw()


class MnistExample(FigureCanvasQTAgg):
    def __init__(self, parent = None):
        super().__init__(Figure())
        self.setParent(parent)
        self.figure.subplots_adjust(bottom=0.2)
        self.axes = self.figure.subplots()
        self.axes.set_xticks([])
        self.axes.set_yticks([])
    
    def show_example(self, example_data, output):
        len_example = len(example_data)
        i = random.randint(0, len_example-1)
        self.axes.imshow(example_data[i][0], cmap='gray', interpolation='none')
        self.axes.set_title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        self.draw()
