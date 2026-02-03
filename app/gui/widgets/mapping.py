from PyQt6 import QtWidgets, QtGui, QtCore

from app.gui.widgets.functions_ui import widgets_switch
from app.gui.widgets.mapping_ui import Ui_MappingPage


class MappingWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MappingPage()
        self.ui.setupUi(self)

        self.setup_validators()

        # Группируем связанные элементы
        self.Frac_inj_elements = [
            # (поле_ввода, подпись)
            (self.ui.leFracHalfLength, self.ui.lblFracHalfLength),
            (self.ui.leMinHorStress, self.ui.lblMinHorStress),
        ]

        # Подключаем сигнал
        self.ui.chkManageAutoFrac.toggled.connect(self.toggle_Frac_inj_fields)

        # Устанавливаем начальное состояние
        self.toggle_Frac_inj_fields(self.ui.chkManageAutoFrac.isChecked())

    def toggle_Frac_inj_fields(self, is_checked):
        """Включает/выключает поля АГРП"""
        for widgets in self.Frac_inj_elements:
            widgets_switch(is_checked, widgets, type_switch='same')

    def setup_validators(self):
        """Проверка полей"""
        int_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"^([1-9]\d{0,5})$"))  # 1-999999
        decimal_fraction_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(0(\.\d{0,3})?|1(\.0{0,3})?)$"))  # 0.000-1.000
        degree_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(360|3[0-5]\d|[12]\d\d|[1-9]?\d)$"))  # 0-360
        float_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(0|[1-9]\d{0,3})(\.\d{1,3})?$"))  # 0.000 - 9999.999
        self.ui.leSizePixel.setValidator(int_validator)
        self.ui.leRadiusInterpolate.setValidator(int_validator)
        self.ui.leMinHorStress.setValidator(degree_validator)
        self.ui.leFracHalfLength.setValidator(float_validator)
        self.ui.leKIN.setValidator(decimal_fraction_validator)

    def get_data(self):
        return {
            "default_size_pixel": int(self.ui.leSizePixel.text()),
            "radius_interpolate": int(self.ui.leRadiusInterpolate.text()),
            "switch_accounting_horwell": self.ui.chkManageHorWells.isChecked(),
            "switch_frac_inj_well": self.ui.chkManageAutoFrac.isChecked(),
            "azimuth_sigma_h_min": int(self.ui.leMinHorStress.text()),
            "l_half_fracture": float(self.ui.leFracHalfLength.text()),
            "KIN": float(self.ui.leKIN.text())
        }

    def check_special_fields(self):
        le_l_half_fracture = self.ui.leFracHalfLength
        l_half_fracture = float(le_l_half_fracture.text().strip())

        if le_l_half_fracture and le_l_half_fracture.isEnabled():
            if not (0 <= l_half_fracture <= 1500):
                le_l_half_fracture.setStyleSheet("border: 1px solid red;")
                QtWidgets.QMessageBox.warning(self, "Ошибка",
                                              "Полудлина трещины АГРП должна быть в диапазоне 0–1500 м")
                return False
            else:
                le_l_half_fracture.setStyleSheet("")  # сброс оформления
                le_l_half_fracture.style().unpolish(le_l_half_fracture)
                le_l_half_fracture.style().polish(le_l_half_fracture)
                le_l_half_fracture.update()
        return True
