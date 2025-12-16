from PyQt6 import QtGui


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
    else:
        if is_checked:
            field.setStyleSheet("background-color: #f0f0f0; color: #a0a0a0;")
        else:
            field.setStyleSheet("")

    # если есть также кнопка в списке
    if len(widgets) > 2:
        button = widgets[2]
        if type_switch == 'same':
            button.setEnabled(is_checked)
        else:
            button.setEnabled(not is_checked)
