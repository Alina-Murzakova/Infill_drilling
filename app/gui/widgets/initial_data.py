from PyQt6 import QtWidgets
from app.gui.widgets.initial_data_ui import Ui_InitialDataPage
from app.input_output.output_functions import get_save_path


class InitialDataWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_InitialDataPage()
        self.ui.setupUi(self)
        self.ui.leSave.setText(get_save_path('АВНС'))

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
