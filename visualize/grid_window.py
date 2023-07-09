from PyQt5 import QtCore, QtGui, QtWidgets


class GridWindow(QtWidgets.QMainWindow):
    """A class to represent the GUI window for visualizing the GridLand environment."""

    def __init__(self, grid):
        """Initialize the GridWindow class."""
        super(GridWindow, self).__init__()
        # Initialize the view and scene
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)

        # Set window properties
        self.setWindowTitle("Grid Visualization")
        self.setGeometry(500, 300, 800, 400)

        # Set the layout for the widget
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.view)
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Initialize the grid
        self.grid = grid
        self.draw_grid()

    def draw_grid(self):
        """Draw the grid to the PyQt5 window."""
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                rect = QtWidgets.QGraphicsRectItem(QtCore.QRectF(j * 40, i * 40, 40, 40))
                rect.setBrush(self.get_brush(self.grid[i][j]))
                self.scene.addItem(rect)

    def update_grid(self, grid):
        """Update the grid when it changes."""
        self.grid = grid
        self.scene.clear()
        self.draw_grid()

    def get_brush(self, value):
        """Get the brush color for a specific cell value."""
        colors = {
            0: QtGui.QColor("white"),  # Ground
            1: QtGui.QColor("black"),  # Wall
            2: QtGui.QColor("green"),  # Start
            3: QtGui.QColor("red"),  # Goal
            4: QtGui.QColor("yellow"),  # Electricity
            5: QtGui.QColor("blue"),  # Button
            6: QtGui.QColor("purple"),  # Car
            7: QtGui.QColor("pink"),  # Fan
            8: QtGui.QColor("orange")  # Agent
        }
        return QtGui.QBrush(colors.get(value, QtGui.QColor("gray")))