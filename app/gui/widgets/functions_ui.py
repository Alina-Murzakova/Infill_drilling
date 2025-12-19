import os

from PyQt6 import QtGui, QtWidgets


def widgets_switch(is_checked, widgets, type_switch='same'):
    """
    Функция для включения виджетов, относительно чек боксов
    type: same - активны, когда чек-бокс включен
         not_same - активны, когда чек-бокс выключен
    """
    field = widgets[0]
    label = widgets[1]
    # Управление доступностью поля ввода
    if type_switch == 'same':
        field.setEnabled(is_checked)
    else:
        field.setEnabled(not is_checked)

    # Управление цветом текста
    palette = label.palette()
    if type_switch == 'same':
        if not is_checked:
            # Серый цвет для неактивных элементов
            palette.setColor(QtGui.QPalette.ColorRole.WindowText,
                             QtGui.QColor(150, 150, 150))
        else:
            # Восстанавливаем стандартный цвет
            palette.setColor(QtGui.QPalette.ColorRole.WindowText,
                             QtGui.QColor(0, 0, 0))
    else:
        if is_checked:
            # Серый цвет для неактивных элементов
            palette.setColor(QtGui.QPalette.ColorRole.WindowText,
                             QtGui.QColor(150, 150, 150))
        else:
            # Восстанавливаем стандартный цвет
            palette.setColor(QtGui.QPalette.ColorRole.WindowText,
                             QtGui.QColor(0, 0, 0))

    label.setPalette(palette)

    # Дополнительно можно установить стиль для полей ввода
    if type_switch == 'same':
        if not is_checked:
            field.setStyleSheet("background-color: #f0f0f0; color: #a0a0a0;")
        else:
            field.setStyleSheet("")
            field.style().unpolish(field)
            field.style().polish(field)
            field.update()
    else:
        if is_checked:
            field.setStyleSheet("background-color: #f0f0f0; color: #a0a0a0;")
        else:
            field.setStyleSheet("")
            field.style().unpolish(field)
            field.style().polish(field)
            field.update()

    # если есть также кнопка в списке
    if len(widgets) > 2:
        button = widgets[2]
        if type_switch == 'same':
            button.setEnabled(is_checked)
        else:
            button.setEnabled(not is_checked)


PATHS_INFO = {
    "data_well_directory": ("file", [".xls", ".xlsx", ".xlsm"]),
    "maps_directory": ("directory", []),
    "path_geo_phys_properties": ("file", [".xls", ".xlsx", ".xlsm"]),
    "path_frac": ("file", [".xls", ".xlsx", ".xlsm"]),
    "path_economy": ("file", [".xls", ".xlsx", ".xlsm"]),
    "save_directory": ("directory", []),
}


def validate_paths(paths, parent=None):
    for key, p in paths.items():
        path = str(p).strip()
        typ, exts = PATHS_INFO[key]

        if path != "":
            if typ == "file":
                if exts and not any(path.lower().endswith(ext) for ext in exts):
                    QtWidgets.QMessageBox.critical(parent, "Ошибка",
                                                   f"Файл '{os.path.basename(path)}' "
                                                   f"имеет недопустимое расширение (.xlsx, .xls или .xlsm).")
                    return False
                elif not os.path.isfile(path):
                    QtWidgets.QMessageBox.critical(parent, "Ошибка",
                                                   f"Файл '{os.path.dirname(path)}' не найден.")
                    return False

            elif typ == "directory":
                if not os.path.isdir(path):
                    QtWidgets.QMessageBox.critical(parent, "Ошибка",
                                                   f"Папка '{os.path.dirname(path)}' не существует.")
                    return False

    return True
