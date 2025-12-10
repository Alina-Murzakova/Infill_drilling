from PyQt6 import QtWidgets, QtGui
from app.gui.widgets.start_ui import Ui_StartPage

import os

path_program = os.getcwd()
logo = "АВНС_v.2.png"


class StartPageWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_StartPage()
        self.ui.setupUi(self)

        logo_path = os.path.abspath(os.path.join(path_program,  "gui", "icons", logo))
        self.ui.lbl_img.setPixmap(QtGui.QPixmap(logo_path))
