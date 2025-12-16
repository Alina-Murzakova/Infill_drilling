from PyQt6 import QtWidgets, QtGui, QtCore
from app.gui.widgets.economy_ui import Ui_EconomyPage
from app.gui.widgets.functions_ui import widgets_switch


class EconomyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_EconomyPage()
        self.ui.setupUi(self)

        self.setup_validators()
        # Кнопка → (line_edit, диалог: "file" или "directory", фильтр для файлов)
        self.ui.btnEconomy.clicked.connect(
            lambda checked, le=self.ui.leEconomy, t="file", f="Excel Files (*.xlsx *.xls *.xlsm);;All Files (*)":
            self.choose_path(le, t, f))

        # Группируем связанные элементы
        self.economy_elements = [
            # (поле_ввода, подпись)
            (self.ui.leStartDate, self.ui.lblStartDate),
            (self.ui.leNumDays, self.ui.lblNumDays),
            (self.ui.leEconomy, self.ui.lblEconomy, self.ui.btnEconomy)
        ]

        # Подключаем сигнал
        self.ui.chkCalcEconomy.toggled.connect(self.toggle_economy_fields)

        # Устанавливаем начальное состояние
        self.toggle_economy_fields(self.ui.chkCalcEconomy.isChecked())

    def toggle_economy_fields(self, is_checked):
        """Включает/выключает поля экономических параметров"""
        for widgets in self.economy_elements:
            widgets_switch(is_checked, widgets, type_switch='same')

    def setup_validators(self):
        """Проверка полей"""
        # day_validator = QtGui.QIntValidator(0, 31)
        day_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"([0-9]|[12][0-9]|3[01])"))
        self.ui.leNumDays.setValidator(day_validator)

    def get_data(self):
        return {
            "switch_economy": self.ui.chkCalcEconomy.isChecked(),
            "start_date": self.ui.leStartDate.date(),
            "day_in_month": int(self.ui.leNumDays.text()),
            "path_economy": self.ui.leEconomy.text()
        }

    def choose_path(self, line_edit, dlg_type, file_filter=""):
        if dlg_type == "file":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать файл", "", file_filter)
        else:
            return
        if path:
            line_edit.setText(path)