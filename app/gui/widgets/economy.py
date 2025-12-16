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
        day_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(30(\.\d{1,3})?|3[01](\.0{1,3})?|[12]?\d(\.\d{1,3})?)$"))  # 0.000-31.000
        self.ui.leNumDays.setValidator(day_validator)

    def get_data(self):
        return {
            "switch_economy": self.ui.chkCalcEconomy.isChecked(),
            "start_date": self.ui.deStartDate.date(),
            "day_in_month": int(self.ui.leNumDays.text())
        }
