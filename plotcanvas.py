from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy
'''
Class necessary to create our custom canvas on QWidgets
'''

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.data = None
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.axis('off')
        # Next line (kind of) removes the white padding around
        fig.subplots_adjust(bottom=0, top= 1, left=0, right=1)
        
        # Blacken everything
        fig.patch.set_facecolor('xkcd:black')
        self.axes.set_facecolor('xkcd:black')
        
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
