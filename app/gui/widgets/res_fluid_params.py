from PyQt6 import QtWidgets, QtGui

from app.gui.widgets.functions_ui import widgets_switch
from app.gui.widgets.res_fluid_params_ui import Ui_ResFluidPage


class ResFluidWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ResFluidPage()
        self.ui.setupUi(self)

        # Группируем связанные элементы
        self.relative_perm_elements = [
            # (поле_ввода, подпись)
            (self.ui.leSor, self.ui.lblSor),
            (self.ui.leSwc, self.ui.lblSwc),
            (self.ui.leFo, self.ui.lblFo),
            (self.ui.leFw, self.ui.lblFw),
            (self.ui.leCoreyOil, self.ui.lblCoreyOil),
            (self.ui.leCoreyWater, self.ui.lblCoreyWater)
        ]

        # Подключаем сигнал
        self.ui.chkRelativePerm.toggled.connect(self.toggle_relative_perm_fields)

        # Устанавливаем начальное состояние
        self.toggle_relative_perm_fields(self.ui.chkRelativePerm.isChecked())

    def toggle_relative_perm_fields(self, is_checked):
        """Включает/выключает поля ОФП"""
        for widgets in self.relative_perm_elements:
            widgets_switch(is_checked, widgets, type_switch='not_same')

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