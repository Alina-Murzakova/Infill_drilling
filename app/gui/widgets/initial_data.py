from PyQt6 import QtWidgets
from app.gui.widgets.initial_data_ui import Ui_InitialDataPage
from app.input_output.output import get_save_path

paths_info = {
    "data_well_directory": ("file", [".xls", ".xlsx", ".xlsm"]),
    "maps_directory": ("directory", []),
    "path_geo_phys_properties": ("file", [".xls", ".xlsx", ".xlsm"]),
    "save_directory": ("directory", []),
}


class InitialDataWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_InitialDataPage()
        self.ui.setupUi(self)
        self.ui.leSave.setText(get_save_path('Infill_drilling'))

        self.setup_file_buttons()

    def get_data(self):
        return {
            "data_well_directory": self.ui.leHistory.text(),
            "maps_directory": self.ui.leMaps.text(),
            "path_geo_phys_properties": self.ui.leProperty.text(),
            "save_directory": self.ui.leSave.text()
        }

    def setup_file_buttons(self):
        # Кнопка → (line_edit, диалог: "file" или "directory", фильтр для файлов)
        buttons = [
            (self.ui.btnHistory, self.ui.leHistory, "file", "Excel Files (*.xlsx *.xls *.xlsm);;All Files (*)"),
            (self.ui.btnMaps, self.ui.leMaps, "directory", ""),
            (self.ui.btnProperty, self.ui.leProperty, "file", "Excel Files (*.xlsx *.xls *.xlsm);;All Files (*)"),
            (self.ui.btnSave, self.ui.leSave, "directory", "")
        ]

        for btn, line_edit, dlg_type, file_filter in buttons:
            btn.clicked.connect(lambda checked, le=line_edit, t=dlg_type, f=file_filter:
                                self.choose_path(le, t, f))

    def choose_path(self, line_edit, dlg_type, file_filter=""):
        if dlg_type == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать файл", "", file_filter)
        elif dlg_type == "directory":
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбрать папку")
        else:
            return
        if path:
            line_edit.setText(path)

    def validate_paths(self):
        import os

        data = self.get_data()

        for key, p in data.items():
            path = data[key].strip()
            typ, exts = paths_info[key]

            if typ == "file":
                if exts and not any(path.lower().endswith(ext) for ext in exts):
                    QtWidgets.QMessageBox.critical(self, "Ошибка",
                                                   f"Файл '{os.path.basename(path)}' "
                                                   f"имеет недопустимое расширение (.xlsx, .xls или .xlsm).")
                    return False
                elif not os.path.isfile(path):
                    QtWidgets.QMessageBox.critical(self, "Ошибка",
                                                   f"Файл '{os.path.dirname(path)}' не найден.")
                    return False

            elif typ == "directory":
                if not os.path.isdir(path):
                    QtWidgets.QMessageBox.critical(self, "Ошибка",
                                                   f"Папка '{os.path.dirname(path)}' не существует.")
                    return False

        return True
