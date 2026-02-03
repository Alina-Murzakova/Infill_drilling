import sys
import os
import time

import pandas as pd

from app.gui.widgets.result import ResultWidget
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import QThread, QObject, pyqtSignal
from loguru import logger
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

from app.gui.main_window_ui import Ui_MainWindow
from app.input_output.output_functions import get_save_path, save_local_parameters
from app.main import run_model
from app.exceptions import CalculationCancelled
from app.gui.widgets.functions_ui import validate_paths
from app.version import APP_NAME, APP_VERSION

path_program = os.getcwd()
icons = [
    "bi--folder-plus.png",
    "bi--layers-half.png",
    "ep--map-location.png",
    "drilling-rig.png",
    "water-drop (1).png",
    "free-icon-dollars-money-bag-50117.png",
    "ep--histogram.png"]

lbl = "lbl.ico"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # Устанавливаем название программы и версию
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")

        # Параметры расчета в формате local_parameters
        self.parameters = None

        # Отдельный поток для расчета
        self.thread = None
        self.worker = None

        # Создаем виджет результатов и добавляем его в stackedWidget
        self.result_widget = ResultWidget()
        # Находим индекс страницы результатов (7-я страница)
        self.ui.stackedWidget.insertWidget(7, self.result_widget)

        # Находим иконки по правильным путям
        for i, item in enumerate(self.ui.listWidget.findItems("*", QtCore.Qt.MatchFlag.MatchWildcard)):
            icon_path = os.path.join(path_program, "gui", "icons", icons[i])
            icon = QtGui.QIcon(icon_path)
            item.setIcon(icon)

        # Иконка приложения
        lbl_path = os.path.join(path_program, "gui", "icons", lbl)
        lbl_app = QtGui.QIcon(lbl_path)
        self.setWindowIcon(lbl_app)

        # Связь меню со страницами
        self.ui.listWidget.currentRowChanged.connect(
            lambda index: self.ui.stackedWidget.setCurrentIndex(index + 1)
        )
        # Снимаем фокус с listWidget, чтобы при запуске была стартовая страница
        self.ui.stackedWidget.setFocus()

        # "О программе"
        self.ui.action.triggered.connect(self.go_to_start_page)

        # "Руководство пользователя"
        self.ui.action_manual.triggered.connect(self.open_user_manual)

        # Запуск расчета
        self.ui.btnCalc.clicked.connect(self.run_calculation)

        # Отмена расчета
        self.ui.btnCancel.clicked.connect(self.cancel_calculation)

    def go_to_start_page(self):
        """Метод для перехода в Файл - О программе"""
        START_PAGE_INDEX = 0
        self.ui.stackedWidget.setCurrentIndex(START_PAGE_INDEX)
        self.ui.listWidget.setCurrentRow(-1)

    def open_user_manual(self):
        """Открытие файла руководства"""
        try:
            # Определяем базовый путь
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))

            # Путь к файлу
            manual_path = os.path.join(base_path, "gui/resources", "manual.docx")

            # Проверяем существование
            if not os.path.exists(manual_path):
                QMessageBox.critical(self, "Ошибка", "Файл руководства не найден!")
                return

            # Просто открываем файл
            # Word автоматически откроет его как обычный документ
            # Пользователь не сможет сохранить изменения в исходный файл
            QDesktopServices.openUrl(QUrl.fromLocalFile(manual_path))

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при открытии файла: {str(e)}")

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

    def collect_all_gui(self) -> dict:
        """Создание словаря с виджетами"""
        return {
            "paths": self.ui.initial_data_page,
            "mapping_params": self.ui.mapping_page,
            "drill_zone_params": self.ui.map_zone_page,
            "well_params": self.ui.well_params_page,
            "res_fluid_params": self.ui.res_fluid_page,
            "economy": self.ui.economy_page,
        }

    @staticmethod
    def convert_to_backend_format(gui_data: dict):
        """Переформирование данных в формат local_parameters"""
        parameters = {
            "paths": {**gui_data["paths"],
                      'path_frac': gui_data["well_params"]["path_frac"],
                      'path_economy': gui_data["economy"]["path_economy"]},

            "switches": {
                "switch_fracList_params": gui_data["well_params"]["switch_fracList_params"],
                "switch_frac_inj_well": gui_data["mapping_params"]["switch_frac_inj_well"],
                "switch_filtration_perm_fact": gui_data["well_params"]["switch_filtration_perm_fact"],
                "switch_economy": gui_data["economy"]["switch_economy"],
                "switch_adaptation_relative_permeability":
                    gui_data["res_fluid_params"]["switch_adaptation_relative_permeability"],
                "switch_wc_from_map": gui_data["well_params"]["switch_wc_from_map"],
                "switch_accounting_horwell": gui_data["mapping_params"]["switch_accounting_horwell"],
                "switch_fix_P_well_init": gui_data["well_params"]["switch_fix_P_well_init"],
            },

            'maps': {"default_size_pixel": gui_data["mapping_params"]["default_size_pixel"],
                     "radius_interpolate": gui_data["mapping_params"]["radius_interpolate"],
                     "azimuth_sigma_h_min": gui_data["mapping_params"]["azimuth_sigma_h_min"],
                     "l_half_fracture": gui_data["mapping_params"]["l_half_fracture"],
                     "KIN": gui_data["mapping_params"]["KIN"]},

            "drill_zones": {**gui_data["drill_zone_params"]},

            'well_params': {
                "general":
                    {"t_p": gui_data["well_params"]["t_p"],
                     "r_w": gui_data["well_params"]["r_w"],
                     "well_efficiency": gui_data["well_params"]["well_efficiency"],
                     "KPPP": gui_data["well_params"]["KPPP"],
                     "skin": gui_data["well_params"]["skin"],
                     "KUBS": gui_data["well_params"]["KUBS"]},

                "fracturing":
                    {"Type_Frac": gui_data["well_params"]["Type_Frac"],
                     "length_FracStage": gui_data["well_params"]["length_FracStage"],
                     "k_f": gui_data["well_params"]["k_f"],
                     "xfr": gui_data["well_params"]["xfr"],
                     "w_f": gui_data["well_params"]["w_f"]},

                "fact_wells_params":
                    {"first_months": gui_data["well_params"]["first_months"],
                     "last_months": gui_data["well_params"]["last_months"],
                     "default_radius_prod": gui_data["well_params"]["default_radius_prod"],
                     "default_radius_inj": gui_data["well_params"]["default_radius_inj"]},

                "proj_wells_params":
                    {"L": gui_data["well_params"]["L"],
                     "min_length": gui_data["well_params"]["min_length"],
                     "buffer_project_wells": gui_data["well_params"]["buffer_project_wells"],
                     "fix_P_well_init": gui_data["well_params"]["fix_P_well_init"],
                     "k": gui_data["well_params"]["k"],
                     "threshold": gui_data["well_params"]["threshold"],
                     'period_calculation': gui_data["well_params"]["period_calculation"]}},

            "reservoir_fluid_properties": {"kv_kh": gui_data["res_fluid_params"]["kv_kh"],
                                           "Swc": gui_data["res_fluid_params"]["Swc"],
                                           "Sor": gui_data["res_fluid_params"]["Sor"],
                                           "Fw": gui_data["res_fluid_params"]["Fw"],
                                           "m1": gui_data["res_fluid_params"]["m1"],
                                           "Fo": gui_data["res_fluid_params"]["Fo"],
                                           "m2": gui_data["res_fluid_params"]["m2"],
                                           "Bw": gui_data["res_fluid_params"]["Bw"]},

            "economy_params": {"day_in_month": gui_data["economy"]["day_in_month"],
                               "start_date": gui_data["economy"]["start_date"]}}
        return parameters

    def run_calculation(self):
        """Запуска расчета"""
        # Проверка на заполнение всех полей
        if not self.check_fields():
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Заполните все поля!")
            return

        gui_data = self.collect_all_gui_data()

        gui_widgets = self.collect_all_gui()
        # дополнительная проверка ряда полей
        for name, widget in gui_widgets.items():
            if hasattr(widget, "check_special_fields"):
                if not widget.check_special_fields():
                    return

        self.parameters = self.convert_to_backend_format(gui_data)
        if not validate_paths(self.parameters["paths"], parent=self):
            return

        self.ui.progressBar.setValue(0)
        self.ui.plainTextEdit.clear()
        self.ui.btnCancel.setEnabled(True)  # активируем кнопку отмены
        self.ui.btnCalc.setEnabled(False)  # блокируем кнопку расчёта

        # Создаём поток и worker в главном потоке
        self.thread = QThread()
        self.worker = Worker(self.parameters)

        # Перенос Worker в другой поток
        self.worker.moveToThread(self.thread)

        # Связываем сигналы (но еще не запускаем)
        self.worker.progress.connect(self.ui.progressBar.setValue)
        self.worker.log.connect(self.ui.plainTextEdit.appendPlainText)
        self.worker.results_ready.connect(self.handle_results)  # для передачи данных в виджет результатов
        self.thread.started.connect(self.worker.run)  # запуск расчета после старта потока
        self.worker.finished.connect(self.thread.quit)  # завершение работы потока (иначе жил бы постоянно)
        self.worker.finished.connect(self.worker.deleteLater)  # удаление Worker (чтобы не было утечек памяти)
        self.thread.finished.connect(self.thread.deleteLater)  # удаляем поток

        # Когда поток полностью завершился — показать QMessageBox
        self.worker.finished.connect(self.finished_calculation)

        # Запуск потока
        self.thread.start()

    def handle_results(self, summary_table: pd.DataFrame, save_directory: str):
        """Обработка результатов расчета"""
        # Передаем данные в виджет результатов
        self.result_widget.set_summary_table(summary_table)
        self.result_widget.set_results_folder(save_directory)

        # Автоматически переключаемся на вкладку результатов
        self.ui.stackedWidget.setCurrentWidget(self.result_widget)
        self.ui.listWidget.setCurrentRow(6)  # 6 соответствует 7-й странице (индексация с 0)

    def finished_calculation(self, success: bool, message: str):
        self.ui.btnCalc.setEnabled(True)
        self.ui.btnCancel.setEnabled(False)
        if success:
            QtWidgets.QMessageBox.information(self, "Готово", message)
        else:
            self.ui.progressBar.setValue(0)
            QtWidgets.QMessageBox.warning(self, "Прервано", message)

    def cancel_calculation(self):
        if self.worker:
            self.ui.btnCancel.setEnabled(False)
            self.worker.stop()

    def check_fields(self):
        all_ok = True
        line_edits = self.ui.stackedWidget.findChildren(QtWidgets.QLineEdit)

        for le in line_edits:
            if not le.text().strip() and le.isEnabled():
                le.setStyleSheet("border: 1px solid red;")
                all_ok = False
            else:
                le.setStyleSheet("")  # сброс оформления
                le.style().unpolish(le)
                le.style().polish(le)
                le.update()
        return all_ok


class Worker(QObject):
    """Объект для выполнения кода"""
    finished = pyqtSignal(bool, str)  # сигнал об окончании расчета
    progress = pyqtSignal(int)  # для прогрессбара
    log = pyqtSignal(str)  # логирование
    results_ready = pyqtSignal(object, str)  # передача результатов (DataFrame, путь)

    def __init__(self, parameters, total_stages=18):
        super().__init__()
        self.parameters = parameters
        self.total_stages = total_stages
        self._is_active = True

        # Храним результаты здесь
        self.summary_table = None
        # Получаем предварительную save_directory из параметров до запуска
        self.save_directory = get_save_path(APP_NAME)

        # Перехват loguru-логов
        self.qt_logger = QtLogger()
        self.qt_logger.log.connect(self.log_message)
        self._sink_id = None
        self._sink_id = logger.add(self.qt_logger, level="INFO")

    def stop(self):
        self._is_active = False
        self.log.emit("Расчёт отменён")

    def is_cancelled(self):
        return not self._is_active

    def log_message(self, msg):
        """Вызывается при каждом logger.info"""
        self.log.emit(msg)

    def run(self):
        """Запуск основной функции"""
        self.log.emit("Расчёт запущен")
        start_time = time.perf_counter()
        try:
            # Передаем Qt-сигналы в run_model
            self.summary_table, self.save_directory = run_model(
                self.parameters,
                self.total_stages,
                progress=self.progress.emit,
                is_cancelled=self.is_cancelled
            )

            # ОТПРАВЛЯЕМ РЕЗУЛЬТАТЫ
            if self.summary_table is not None:
                self.results_ready.emit(self.summary_table, self.save_directory)

            elapsed = time.perf_counter() - start_time
            self.finished.emit(True, f"Расчёт успешно завершён\nВремя: {format_time(elapsed)}")

        except CalculationCancelled:
            message = "Расчёт отменён пользователем"
            logger.info(message)
            self.finished.emit(False, message)

        except Exception as e:
            message = f"⚠ Ошибка расчёта:\n{str(e)}"
            logger.error(message)
            # Сохраняем логи при ошибке
            self._save_error_to_file(e, self.parameters)

            self.finished.emit(False, message)

        finally:
            if self._sink_id is not None:
                logger.remove(self._sink_id)

    def _save_error_to_file(self, error, parameters):
        """Сохраняет информацию об ошибке в файл"""
        try:
            if self.save_directory:
                from pathlib import Path
                from datetime import datetime
                import traceback
                log_path = Path(self.save_directory) / ".debug" / "error.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)

                content = f"Время ошибки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                content += f"Тип ошибки: {type(error).__name__}\n"
                content += f"Сообщение: {str(error)}\n\n"
                content += "Трассировка:\n"
                content += traceback.format_exc()

                # Записываем в файл (дописываем, если файл существует)
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write(content)
                    f.write("=" * 60 + "\n\n")

                save_local_parameters(parameters, f"{self.save_directory}/.debug/local_parameters.py")
                logger.info(f"Информация об ошибке сохранена в: {log_path}")

        except Exception as log_error:
            logger.error(f"Не удалось сохранить информацию об ошибке: {log_error}")


class QtLogger(QObject):
    log = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def write(self, msg):
        msg = msg.strip()
        if msg:  # фильтруем пустые строки
            self.log.emit(msg)

    def flush(self):
        pass


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} сек"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)} мин {int(s)} сек"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)} ч {int(m)} мин {int(s)} сек"


if __name__ == "__main__":
    app = QApplication(sys.argv)  # Настройки компьютера

    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    app.setStyle("Fusion")  # Изменяет системный стиль
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
