import sys
import os

from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtWidgets import QApplication

from app.gui.main_window_ui import Ui_MainWindow
from app.main import run_model

path_program = os.getcwd()
icons = [
    "bi--folder-plus.png",
    "bi--layers-half.png",
    "ep--map-location.png",
    "drilling-rig.png",
    "water-drop (1).png",
    "free-icon-dollars-money-bag-50117.png",
    "ep--histogram.png"]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Параметры расчета в формате local_parameters
        self.main_parameters = None
        self.constants = None

        # Находим иконки по правильным путям
        for i, item in enumerate(self.ui.listWidget.findItems("*", QtCore.Qt.MatchFlag.MatchWildcard)):
            icon_path = os.path.join(path_program, "gui", "icons", icons[i])
            icon = QtGui.QIcon(icon_path)
            item.setIcon(icon)

        # Связь меню со страницами
        self.ui.listWidget.currentRowChanged.connect(
            lambda index: self.ui.stackedWidget.setCurrentIndex(index + 1)
        )
        # Снимаем фокус с listWidget, чтобы при запуске была стартовая страница
        self.ui.stackedWidget.setFocus()

        # "О программе"
        self.ui.action.triggered.connect(self.go_to_start_page)

        # Запуск расчета
        self.ui.btnCalc.clicked.connect(self.run_calculation)

    def go_to_start_page(self):
        """Метод для перехода в Файл - О программе"""
        START_PAGE_INDEX = 0
        self.ui.stackedWidget.setCurrentIndex(START_PAGE_INDEX)
        self.ui.listWidget.setCurrentRow(-1)

    def collect_all_gui_data(self) -> dict:
        """Сбор данных с интерфейса"""
        return {
            "paths": self.ui.initial_data_page.get_data(),
            "mapping_params": self.ui.mapping_page.get_data(),
            "drill_zone_params": self.ui.map_zone_page.get_data(),
            "well_params": self.ui.well_params_page.get_data(),
            "res_fluid_params": self.ui.res_fluid_page.get_data(),
            "economy": self.ui.economy_page.get_data(),
        }

    @staticmethod
    def convert_to_backend_format(gui_data: dict) -> tuple:
        """Переформирование данных в формат local_parameters"""
        main_parameters = {
            "paths": {**gui_data["paths"],
                      'path_frac': gui_data["well_params"]["path_frac"],
                      'path_economy': gui_data["economy"]["start_date"]},

            "parameters_calculation": {
                **gui_data["drill_zone_params"],
                "period_calculation": gui_data["well_params"]["period_calculation"],
            },

            "switches": {
                "switch_avg_frac_params": gui_data["well_params"]["switch_avg_frac_params"],
                "switch_fracture": gui_data["mapping_params"]["switch_fracture"],
                "switch_permeability_fact": gui_data["well_params"]["switch_permeability_fact"],
                "switch_economy": gui_data["economy"]["switch_economy"],
                "switch_adaptation_relative_permeability":
                    gui_data["res_fluid_params"]["switch_adaptation_relative_permeability"],
                "water_cut_map": gui_data["well_params"]["water_cut_map"],
                "accounting_GS": gui_data["mapping_params"]["accounting_GS"],
                "fix_P_delta": gui_data["well_params"]["fix_P_delta"],
            },

            "well_params": {
                "L": gui_data["well_params"]["L"],
                "xfr": gui_data["well_params"]["xfr"],
                "w_f": gui_data["well_params"]["w_f"],
                "Type_Frac": gui_data["well_params"]["Type_Frac"],
                "length_FracStage": gui_data["well_params"]["length_FracStage"],
                "k_f": gui_data["well_params"]["k_f"],
                "t_p": gui_data["well_params"]["t_p"],
                "r_w": gui_data["well_params"]["r_w"],
                "P_well_init": gui_data["well_params"]["P_well_init"],
                "well_efficiency": gui_data["well_params"]["well_efficiency"],
                "azimuth_sigma_h_min": gui_data["mapping_params"]["azimuth_sigma_h_min"],
                "l_half_fracture": gui_data["mapping_params"]["l_half_fracture"],
            },
        }
        constants = {
            "load_data_param": {
                "default_size_pixel": gui_data["mapping_params"]["default_size_pixel"],
                "first_months": gui_data["well_params"]["first_months"],
                "last_months": gui_data["well_params"]["last_months"],
                "radius_interpolate": gui_data["mapping_params"]["radius_interpolate"],
            },
            "default_coefficients": {
                "KPPP": gui_data["well_params"]["KPPP"],
                "skin": gui_data["well_params"]["skin"],
                "KUBS": gui_data["well_params"]["KUBS"],
                "KIN": gui_data["mapping_params"]["KIN"],
            },
            "default_well_params": {
                "kv_kh": gui_data["res_fluid_params"]["kv_kh"],
                "Swc": gui_data["res_fluid_params"]["Swc"],
                "Sor": gui_data["res_fluid_params"]["Sor"],
                "Fw": gui_data["res_fluid_params"]["Fw"],
                "m1": gui_data["res_fluid_params"]["m1"],
                "Fo": gui_data["res_fluid_params"]["Fo"],
                "m2": gui_data["res_fluid_params"]["m2"],
                "Bw": gui_data["res_fluid_params"]["Bw"],
                "default_radius": gui_data["well_params"]["default_radius"],
                "default_radius_inj": gui_data["well_params"]["default_radius_inj"],
            },
            "default_project_well_params": {
                "buffer_project_wells": gui_data["well_params"]["buffer_project_wells"],
                "day_in_month": gui_data["economy"]["day_in_month"],
                "threshold": gui_data["well_params"]["threshold"],
                "k": gui_data["well_params"]["k"],
                "min_length": gui_data["well_params"]["min_length"],
                "start_date": gui_data["economy"]["start_date"],
            },
        }

        return main_parameters, constants

    def run_calculation(self):
        """Запуска расчета"""
        # Проверка на заполнение всех полей
        if not self.check_fields():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Заполните все поля!")
            return

        # if not self.ui.initial_data_page.validate_paths():
        #     return

        gui_data = self.collect_all_gui_data()
        self.main_parameters, self.constants = self.convert_to_backend_format(gui_data)
        print("Кнопка нажата")
        run_model(self.main_parameters, self.constants)

    def check_fields(self):
        all_ok = True

        line_edits = self.ui.stackedWidget.findChildren(QtWidgets.QLineEdit)

        for le in line_edits:
            if not le.text().strip() and le.isEnabled():
                le.setStyleSheet("border: 1px solid red;")
                all_ok = False
            else:
                le.setStyleSheet("")  # сброс оформления
        return all_ok


if __name__ == "__main__":
    app = QApplication(sys.argv)  # Настройки компьютера
    app.setStyle("Fusion")  # Изменяет системный стиль
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
