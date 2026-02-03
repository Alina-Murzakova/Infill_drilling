from PyQt6 import QtWidgets, QtGui, QtCore

from app.gui.widgets.functions_ui import widgets_switch
from app.gui.widgets.res_fluid_params_ui import Ui_ResFluidPage


class ResFluidWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ResFluidPage()
        self.ui.setupUi(self)

        self.setup_validators()

        # Группируем связанные элементы
        self.resfluid_elements = [
            # (поле_ввода, подпись)
            (self.ui.leSor, self.ui.lblSor),
            (self.ui.leSwc, self.ui.lblSwc),
            (self.ui.leFo, self.ui.lblFo),
            (self.ui.leFw, self.ui.lblFw),
            (self.ui.leCoreyOil, self.ui.lblCoreyOil),
            (self.ui.leCoreyWater, self.ui.lblCoreyWater)
        ]

        # Подключаем сигнал
        self.ui.chkRelativePerm.toggled.connect(self.toggle_economy_fields)

        # Устанавливаем начальное состояние
        self.toggle_economy_fields(self.ui.chkRelativePerm.isChecked())

    def toggle_economy_fields(self, is_checked):
        """Включает/выключает поля экономических параметров"""
        for widgets in self.resfluid_elements:
            widgets_switch(is_checked, widgets, type_switch='not_same')

    def setup_validators(self):
        """Проверка полей"""
        decimal_fraction_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(0(\.\d{0,3})?|1(\.0{0,3})?)$"))  # 0.000-1.000
        power_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(10(\.0{0,3})?|[0-9](\.\d{0,3})?)$"))  # 0.000-10.000
        specific_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(1(\.([0-4]\d{0,2}|5(\.0{0,2})?)?)?|0(\.\d{0,3})?)$"))  # 0.000-1.500

        self.ui.leSor.setValidator(decimal_fraction_validator)
        self.ui.leSwc.setValidator(decimal_fraction_validator)
        self.ui.leFo.setValidator(decimal_fraction_validator)
        self.ui.leFw.setValidator(decimal_fraction_validator)
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

    def check_special_fields(self):
        dict_fields = {self.ui.lblFo.text(): self.ui.leFo,
                       self.ui.lblFw.text(): self.ui.leFw,
                       self.ui.lblAnisotropy.text(): self.ui.leAnisotropy}

        for name, field in dict_fields.items():
            param = float(field.text().strip())

            if field and field.isEnabled():
                if param < 0.001:
                    field.setStyleSheet("border: 1px solid red;")
                    QtWidgets.QMessageBox.warning(self, "Ошибка",
                                                  f"Значение параметра '{name}' должно быть выше 0.000!")
                    return False
                else:
                    field.setStyleSheet("")  # сброс оформления
                    field.style().unpolish(field)
                    field.style().polish(field)
                    field.update()
        return True
