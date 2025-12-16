from PyQt6 import QtWidgets, QtGui, QtCore
from app.gui.widgets.well_params_ui import Ui_WellParamsPage


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
            "P_well_init": float(self.ui.lePwell.text()),
            "water_cut_map": self.get_water_cut_source(),
            "k": int(self.ui.leNumNearWells.text()),
            "threshold": float(self.ui.leThreshold.text()),
            "period_calculation": int(self.ui.leForecastPeriod.text()),
        }

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
