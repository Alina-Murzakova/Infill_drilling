from PyQt6 import QtWidgets, QtGui, QtCore
from app.gui.widgets.res_fluid_params_ui import Ui_ResFluidPage


class ResFluidWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ResFluidPage()
        self.ui.setupUi(self)

        self.setup_validators()

    def setup_validators(self):
        """Проверка полей"""
        float_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(0(\.\d{0,3})?|1(\.0{0,3})?)$"))  # 0.000-1.000
        power_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(10(\.0{0,3})?|[0-9](\.\d{0,3})?)$"))  # 0.000-10.000
        specific_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(1(\.([0-4]\d{0,2}|5(\.0{0,2})?)?)?|0(\.\d{0,3})?)$"))  # 0.000-1.500

        self.ui.leSor.setValidator(float_validator)
        self.ui.leSwc.setValidator(float_validator)
        self.ui.leFo.setValidator(float_validator)
        self.ui.leFw.setValidator(float_validator)
        self.ui.leCoreyOil.setValidator(power_validator)
        self.ui.leCoreyWater.setValidator(power_validator)
        self.ui.leBw.setValidator(specific_validator)
        self.ui.leAnisotropy.setValidator(specific_validator)

    def get_data(self):
        return {
            "switch_adaptation_relative_permeability": self.ui.chkRelativePerm.isChecked(),
            "Sor": float(self.ui.leSor.text()),
            "Swc": float(self.ui.leSwc.text()),
            "Fo": float(self.ui.leFo.text()),
            "Fw": float(self.ui.leFw.text()),
            "m2": float(self.ui.leCoreyOil.text()),
            "m1": float(self.ui.leCoreyWater.text()),
            "Bw": float(self.ui.leBw.text()),
            "kv_kh": float(self.ui.leAnisotropy.text()),
        }
