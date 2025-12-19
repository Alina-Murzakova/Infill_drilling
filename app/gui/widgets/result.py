from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtCore import Qt
from app.gui.widgets.result_ui import Ui_ResultPage
import pandas as pd
import os
import sys


class ResultWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ResultPage()
        self.ui.setupUi(self)

        # Модель для таблицы
        self.model = QtGui.QStandardItemModel()
        self.ui.tableSummary.setModel(self.model)

        # Настройка таблицы
        self.ui.tableSummary.horizontalHeader().setStretchLastSection(True)
        self.ui.tableSummary.setAlternatingRowColors(True)

        # Связываем кнопку
        self.ui.btnOpenResult.clicked.connect(self.open_results_folder)

        # Путь к папке с результатами
        self.results_folder = None

    def set_summary_table(self, df: pd.DataFrame):
        """Установка DataFrame в таблицу с автоопределением ширины"""
        if df is None or df.empty:
            return

        self.model.clear()
        self.model.setHorizontalHeaderLabels(df.columns.tolist())

        # Максимальная ширина столбца
        max_column_width = 150
        min_column_width = 50

        # Заполняем данными
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QtGui.QStandardItem(str(df.iat[row, col]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.model.setItem(row, col, item)

        # Автонастройка ширины с ограничениями
        self.ui.tableSummary.resizeColumnsToContents()

        # Ограничиваем максимальную ширину
        for col in range(self.model.columnCount()):
            current_width = self.ui.tableSummary.columnWidth(col)
            if current_width > max_column_width:
                self.ui.tableSummary.setColumnWidth(col, max_column_width)
            elif current_width < min_column_width:
                self.ui.tableSummary.setColumnWidth(col, min_column_width)

        # Включаем перенос текста
        self.ui.tableSummary.resizeRowsToContents()

        # Настройка заголовков
        header = self.ui.tableSummary.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)

    def set_results_folder(self, folder_path: str):
        """Установка пути к папке с результатами"""
        self.results_folder = folder_path

    def open_results_folder(self):
        """Открытие папки с результатами"""
        if self.results_folder and os.path.exists(self.results_folder):
            try:
                if sys.platform == 'win32':
                    os.startfile(self.results_folder)
                elif sys.platform == 'darwin':  # macOS
                    import subprocess
                    subprocess.Popen(['open', self.results_folder])
                else:  # Linux
                    import subprocess
                    subprocess.Popen(['xdg-open', self.results_folder])
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Ошибка",
                    f"Не удалось открыть папку:\n{str(e)}"
                )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Папка не найдена",
                "Папка с результатами не найдена или расчет еще не выполнен."
            )