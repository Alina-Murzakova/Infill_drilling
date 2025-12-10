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
        int_validator = QtGui.QIntValidator(0, 999999)
        float_validator = QtGui.QDoubleValidator(0.0, 1.0, 3)
        float_validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)  # запись без экспоненты
        self.ui.leSizePixel.setValidator(int_validator)
        self.ui.leRadiusInterpolate.setValidator(int_validator)
        self.ui.leMinHorStress.setValidator(int_validator)
        self.ui.leFracHalfLength.setValidator(int_validator)
        self.ui.leKIN.setValidator(float_validator)

    def get_data(self):
        return {
            "default_size_pixel": int(self.ui.leSizePixel.text()),
            "radius_interpolate": int(self.ui.leRadiusInterpolate.text()),
            "accounting_GS": self.ui.chkManageHorWells.isChecked(),
            "switch_fracture": self.ui.chkManageAutoFrac.isChecked(),
            "azimuth_sigma_h_min": self.ui.leMinHorStress.text(),
            "l_half_fracture": int(self.ui.leFracHalfLength.text()),
            "KIN": float(self.ui.leKIN.text())
        }