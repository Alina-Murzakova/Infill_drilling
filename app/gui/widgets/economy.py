from PyQt6 import QtWidgets, QtGui, QtCore
from app.gui.widgets.economy_ui import Ui_EconomyPage


class EconomyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_EconomyPage()
        self.ui.setupUi(self)

        self.setup_validators()

    def setup_validators(self):
        """Проверка полей"""
        # day_validator = QtGui.QIntValidator(0, 31)
        day_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"([0-9]|[12][0-9]|3[01])"))
        self.ui.leNumDays.setValidator(day_validator)

    def get_data(self):
        return {
            "switch_economy": self.ui.chkCalcEconomy.isChecked(),
            "start_date": self.ui.deStartDate.date(),
            "day_in_month": int(self.ui.leNumDays.text())
        }