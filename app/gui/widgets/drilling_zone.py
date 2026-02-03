from PyQt6 import QtWidgets, QtGui, QtCore
from app.gui.widgets.drilling_zone_ui import Ui_DrillingZonePage


class DrillingZoneWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_DrillingZonePage()
        self.ui.setupUi(self)

        self.setup_validators()

    def setup_validators(self):
        """Проверка полей"""
        int_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"^([1-9]\d{0,5})$"))  # 1-999999
        percent_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"^(100|[1-9]?\d)$"))  # 0-100
        self.ui.lePercentTop.setValidator(percent_validator)

        self.ui.lePercentTop.setValidator(percent_validator)
        self.ui.leMinRadius.setValidator(int_validator)
        self.ui.leSensitivity.setValidator(percent_validator)
        self.ui.leProfitCumOil.setValidator(int_validator)

    def get_data(self):
        return {
            "percent_top": int(self.ui.lePercentTop.text()),
            "min_radius": int(self.ui.leMinRadius.text()),
            "sensitivity_quality_drill": int(self.ui.leSensitivity.text()),
            "init_profit_cum_oil": int(self.ui.leProfitCumOil.text()),
        }
