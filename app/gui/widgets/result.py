from PyQt6 import QtWidgets
from app.gui.widgets.result_ui import Ui_ResultPage


class ResultWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ResultPage()
        self.ui.setupUi(self)