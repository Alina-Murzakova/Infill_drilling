from PyQt6 import QtWidgets
from app.gui.widgets.res_fluid_params_ui import Ui_ResFluidPage


class ResFluidWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ResFluidPage()
        self.ui.setupUi(self)

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
