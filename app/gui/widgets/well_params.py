from PyQt6 import QtWidgets
from app.gui.widgets.well_params_ui import Ui_WellParamsPage


class WellParamsWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_WellParamsPage()
        self.ui.setupUi(self)

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
