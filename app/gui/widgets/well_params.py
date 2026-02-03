from PyQt6 import QtWidgets, QtGui, QtCore
from app.gui.widgets.well_params_ui import Ui_WellParamsPage
from app.gui.widgets.functions_ui import widgets_switch


class WellParamsWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_WellParamsPage()
        self.ui.setupUi(self)

        self.setup_validators()

    def setup_validators(self):
        """Проверка полей"""
        int_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"^([1-9]\d{0,5})$"))  # 1-999999
        int_validator_zero = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"^(0|[1-9]\d{0,5})$"))  # 0-999999
        decimal_fraction_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(0(\.\d{0,3})?|1(\.0{0,3})?)$"))  # 0.000-1.000
        skin_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^-?(0|[1-9]\d{0,1})(\.\d{0,3})?$"))  # -99.000-(+99.000)

        # ---------- Общие ----------
        self.ui.leWellRadius.setValidator(decimal_fraction_validator)
        self.ui.leRampUpTime.setValidator(int_validator)
        self.ui.leWellEfficiency.setValidator(decimal_fraction_validator)
        self.ui.leKUBS.setValidator(decimal_fraction_validator)
        self.ui.leKPPP.setValidator(decimal_fraction_validator)
        self.ui.leSkin.setValidator(skin_validator)

        # ---------- ГРП ----------
        self.ui.leLenStage.setValidator(int_validator)
        self.ui.leXfr.setValidator(int_validator)
        self.ui.leWf.setValidator(int_validator)
        self.ui.lePermProppant.setValidator(int_validator)

        # ---------- Фактический фонд ----------
        self.ui.leFirstMonths.setValidator(int_validator)
        self.ui.leLastMonths.setValidator(int_validator)
        self.ui.leMinBufferProd.setValidator(int_validator_zero)
        self.ui.leMiBufferInj.setValidator(int_validator_zero)

        # ---------- Проектный фонд ----------
        self.ui.leMaxLenWell.setValidator(int_validator_zero)
        self.ui.leMinLenHor.setValidator(int_validator_zero)
        self.ui.leProjectBuffer.setValidator(int_validator_zero)
        self.ui.lePwell.setValidator(int_validator)
        self.ui.leNumNearWells.setValidator(int_validator)
        self.ui.leThreshold.setValidator(int_validator_zero)
        self.ui.leForecastPeriod.setValidator(int_validator_zero)
        # Кнопка → (line_edit, диалог: "file" или "directory", фильтр для файлов)
        self.ui.btnFrac.clicked.connect(
            lambda checked, le=self.ui.leFrac, t="file", f="Excel Files (*.xlsx *.xls *.xlsm);;All Files (*)":
            self.choose_path(le, t, f))

        # Группируем связанные элементы
        self.Frac_elements = [
            # (поле_ввода, подпись)
            (self.ui.leLenStage, self.ui.lblLenStage),
            (self.ui.leXfr, self.ui.lblXfr),
            (self.ui.leWf, self.ui.lblWf),
            (self.ui.lePermProppant, self.ui.lblPermProppant)
        ]
        self.btnFrac_elements = [(self.ui.leFrac, self.ui.lblFrac, self.ui.btnFrac)]
        self.P_well_elements = [(self.ui.lePwell, self.ui.lblPwell)]

        # Подключаем сигнал
        self.ui.chkUseFracSheet.toggled.connect(self.toggle_Frac_fields)
        self.ui.chkSetPwell.toggled.connect(self.toggle_P_well_fields)

        # Устанавливаем начальное состояние
        self.toggle_Frac_fields(self.ui.chkUseFracSheet.isChecked())
        self.toggle_P_well_fields(self.ui.chkSetPwell.isChecked())

    def toggle_Frac_fields(self, is_checked):
        """Включает/выключает поля параметров ГРП"""
        for widgets in self.Frac_elements:
            widgets_switch(is_checked, widgets, type_switch='not_same')

        """Включает/выключает кнопку  "выбор файла" для фрак листов"""
        for widgets in self.btnFrac_elements:
            widgets_switch(is_checked, widgets, type_switch='same')

    def toggle_P_well_fields(self, is_checked):
        """Включает/выключает поля забойного давления"""
        for widgets in self.P_well_elements:
            widgets_switch(is_checked, widgets, type_switch='same')

    def get_data(self):
        return {
            # ---------- Общие ----------
            "r_w": float(self.ui.leWellRadius.text()),
            "t_p": float(self.ui.leRampUpTime.text()),
            "well_efficiency": float(self.ui.leWellEfficiency.text()),
            "KUBS": float(self.ui.leKUBS.text()),
            "KPPP": float(self.ui.leKPPP.text()),
            "skin": float(self.ui.leSkin.text()),

            # ---------- ГРП ----------
            "Type_Frac": self.get_frac_type(),
            "switch_fracList_params": self.ui.chkUseFracSheet.isChecked(),
            "length_FracStage": float(self.ui.leLenStage.text()),
            "xfr": float(self.ui.leXfr.text()),
            "w_f": float(self.ui.leWf.text()),
            "k_f": float(self.ui.lePermProppant.text()),
            "path_frac": self.ui.leFrac.text(),

            # ---------- Фактический фонд ----------
            "first_months": int(self.ui.leFirstMonths.text()),
            "last_months": int(self.ui.leLastMonths.text()),
            "default_radius_prod": float(self.ui.leMinBufferProd.text()),
            "default_radius_inj": float(self.ui.leMiBufferInj.text()),
            "switch_filtration_perm_fact": self.ui.chkFiltrationPerm.isChecked(),

            # ---------- Проектный фонд ----------
            "L": int(self.ui.leMaxLenWell.text()),
            "min_length": int(self.ui.leMinLenHor.text()),
            "buffer_project_wells": int(self.ui.leProjectBuffer.text()),
            "switch_fix_P_well_init": self.ui.chkSetPwell.isChecked(),
            "fix_P_well_init": float(self.ui.lePwell.text()) if self.ui.lePwell.text() else 0.0,
            "switch_wc_from_map": self.get_water_cut_source(),
            "k": int(self.ui.leNumNearWells.text()),
            "threshold": float(self.ui.leThreshold.text()),
            "period_calculation": int(self.ui.leForecastPeriod.text()),
        }

    def check_special_fields(self):
        dict_fields = {self.ui.lblWellRadius.text(): self.ui.leWellRadius,
                       self.ui.lblKUBS.text(): self.ui.leKUBS,
                       self.ui.lblKPPP.text(): self.ui.leKPPP,
                       self.ui.lblSkin.text(): self.ui.leSkin}

        for name, field in dict_fields.items():
            param = float(field.text().strip())

            if field and field.isEnabled():
                if name == "Доп. сопротивление притоку":
                    if self.get_frac_type() is None and int(self.ui.leMaxLenWell.text()) == 0:
                        if param < -3:
                            field.setStyleSheet("border: 1px solid red;")
                            QtWidgets.QMessageBox.warning(self, "Ошибка",
                                                          f"Значение параметра '{name}' должно быть >= -3!")
                            return False
                    else:
                        if param < 0:
                            field.setStyleSheet("border: 1px solid red;")
                            QtWidgets.QMessageBox.warning(self, "Ошибка",
                                                          f"Значение параметра '{name}' должно быть >= 0!")
                            return False
                else:
                    if param < 0.001:
                        field.setStyleSheet("border: 1px solid red;")
                        QtWidgets.QMessageBox.warning(self, "Ошибка",
                                                      f"Значение параметра '{name}' должно быть выше 0.000!")
                        return False
                field.setStyleSheet("")  # сброс оформления
                field.style().unpolish(field)
                field.style().polish(field)
                field.update()
        return True

    def choose_path(self, line_edit, dlg_type, file_filter=""):
        if dlg_type == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать файл", "", file_filter)
        else:
            return
        if path:
            line_edit.setText(path)

    # ---------------- Вспомогательные методы ----------------
    def get_frac_type(self):
        if self.ui.rbtnNoFrac.isChecked():
            return None
        elif self.ui.rbtnFrac.isChecked():
            return "ГРП"
        elif self.ui.rbtnMultiFrac.isChecked():
            return "МГРП"

    def get_water_cut_source(self) -> bool:
        if self.ui.rbtnWaterCutMap.isChecked():
            return True
        elif self.ui.rbtnWaterCutWells.isChecked():
            return False
