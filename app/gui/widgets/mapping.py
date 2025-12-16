from PyQt6 import QtWidgets, QtGui, QtCore
from app.gui.widgets.mapping_ui import Ui_MappingPage


class MappingWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MappingPage()
        self.ui.setupUi(self)

        self.setup_validators()

    def setup_validators(self):
        """Проверка полей"""
        int_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"^([1-9]\d{0,5})$"))  # 1-999999
        float_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(0(\.\d{0,3})?|1(\.0{0,3})?)$"))  # 0.000-1.000
        degree_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(360|3[0-5]\d|[12]\d\d|[1-9]?\d)$"))  # 0-360
        double_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^[1-9]\d{0,5}(\.\d{1,3})?$"))  # 1.000 - 999999.000
        self.ui.leSizePixel.setValidator(int_validator)
        self.ui.leRadiusInterpolate.setValidator(int_validator)
        self.ui.leMinHorStress.setValidator(degree_validator)
        self.ui.leFracHalfLength.setValidator(double_validator)
        self.ui.leKIN.setValidator(float_validator)

    def get_data(self):
        return {
            "default_size_pixel": int(self.ui.leSizePixel.text()),
            "radius_interpolate": int(self.ui.leRadiusInterpolate.text()),
            "accounting_GS": self.ui.chkManageHorWells.isChecked(),
            "switch_fracture": self.ui.chkManageAutoFrac.isChecked(),
            "azimuth_sigma_h_min": self.ui.leMinHorStress.text(),
            "l_half_fracture": float(self.ui.leFracHalfLength.text()),
            "KIN": float(self.ui.leKIN.text())
        }