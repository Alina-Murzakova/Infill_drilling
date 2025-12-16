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
        int_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"^(0|[1-9]\d{0,5})$"))  # 0-999999
        float_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(0(\.\d{0,3})?|1(\.0{0,3})?)$"))  # 0.000-1.000
        pressure_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(1000(\.0{0,3})?|[1-9]\d{0,2}(\.\d{1,3})?)$"))  # 1.0-1000.0
        skin_validator = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"^(-?(100(\.0{0,3})?|[0-9]?\d(\.\d{0,3})?))$"))  # -100.000-(+100.000)

        # ---------- Общие ----------
        self.ui.leWellRadius.setValidator(float_validator)
        self.ui.leRampUpTime.setValidator(int_validator)
        self.ui.leWellEfficiency.setValidator(float_validator)
        self.ui.leKUBS.setValidator(float_validator)
        self.ui.leKPPP.setValidator(float_validator)
        self.ui.leSkin.setValidator(skin_validator)

        # ---------- ГРП ----------
        self.ui.leLenStage.setValidator(int_validator)
        self.ui.leXfr.setValidator(int_validator)
        self.ui.leWf.setValidator(int_validator)
        self.ui.lePermProppant.setValidator(int_validator)

        # ---------- Фактический фонд ----------
        self.ui.leFirstMonths.setValidator(int_validator)
        self.ui.leLastMonths.setValidator(int_validator)
        self.ui.leMinBufferProd.setValidator(int_validator)
        self.ui.leMiBufferInj.setValidator(int_validator)

        # ---------- Проектный фонд ----------
        self.ui.leMaxLenWell.setValidator(int_validator)
        self.ui.leMinLenHor.setValidator(int_validator)
        self.ui.leProjectBuffer.setValidator(int_validator)
        self.ui.lePwell.setValidator(pressure_validator)
        self.ui.leNumNearWells.setValidator(int_validator)
        self.ui.leThreshold.setValidator(int_validator)
        self.ui.leForecastPeriod.setValidator(int_validator)
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
            "switch_avg_frac_params": self.ui.chkUseFracSheet.isChecked(),
            "length_FracStage": float(self.ui.leLenStage.text()),
            "xfr": float(self.ui.leXfr.text()),
            "w_f": float(self.ui.leWf.text()),
            "k_f": float(self.ui.lePermProppant.text()),
            "path_frac": self.ui.leFrac.text(),

            # ---------- Фактический фонд ----------
            "first_months": int(self.ui.leFirstMonths.text()),
            "last_months": int(self.ui.leLastMonths.text()),
            "default_radius": float(self.ui.leMinBufferProd.text()),
            "default_radius_inj": float(self.ui.leMiBufferInj.text()),
            "switch_permeability_fact": self.ui.chkFiltrationPerm.isChecked(),

            # ---------- Проектный фонд ----------
            "L": int(self.ui.leMaxLenWell.text()),
            "min_length": int(self.ui.leMinLenHor.text()),
            "buffer_project_wells": int(self.ui.leProjectBuffer.text()),
            "fix_P_delta": self.ui.chkSetPwell.isChecked(),
            "P_well_init": self.ui.lePwell.text(),
            "water_cut_map": self.get_water_cut_source(),
            "k": int(self.ui.leNumNearWells.text()),
            "threshold": float(self.ui.leThreshold.text()),
            "period_calculation": int(self.ui.leForecastPeriod.text()),
        }

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
