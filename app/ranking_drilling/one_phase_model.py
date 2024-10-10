"""Теория Баклея-Леверетта - однофазная модель притока"""
from scipy.optimize import bisect


def get_one_phase_model(fluid_params, reservoir_params, Swc, Sor, Fw, m1, Fo, m2, Bw):
    """Расчет общих физических параметров системы"""
    # Выделение необходимых параметров пластовой жидкости
    mu_w, mu_o, Bo, c_o, c_w = list(map(lambda key: fluid_params[key], ['mu_w', 'mu_o', 'Bo', 'c_o', 'c_w']))
    # Выделение необходимых параметров пласта
    f_w, c_r = list(map(lambda key: reservoir_params[key], ['f_w', 'c_r']))

    # Водонасыщенность в области бурения
    Sw = get_sw(mu_w, mu_o, Bo, Bw, f_w, Fw, m1, Fo, m2, Swc, Sor)
    # Эффективная вязкость жидкости
    mu = get_mu(Sw, mu_w, mu_o, Fw, m1, Fo, m2, Swc, Sor)
    # Общая сжимаемость системы
    c_t = c_o * (1 - Sw) + c_w * Sw + c_r
    # Эффективный объемный коэффициент расширения жидкости
    B = get_b(Bo, Bw, f_w)

    return mu, c_t, B


def get_sw(mu_w, mu_o, Bo, Bw, f_w, Fw, m1, Fo, m2, Swc, Sor):
    """
    Водонасыщенность пласта от обводненности f_w (решение обратной задачи делением пополам)
    """
    Sw_min = Swc  # нижняя граница интервала поиска решения
    Sw_max = 1 - Sor  # верхняя граница интервала поиска решения
    # проверка краевых значений
    if f_w <= get_f_w(mu_w, mu_o, Bo, Bw, Sw_min, Fw, m1, Fo, m2, Swc, Sor):
        Sw = Sw_min
    elif f_w >= get_f_w(mu_w, mu_o, Bo, Bw, Sw_max, Fw, m1, Fo, m2, Swc, Sor):
        Sw = Sw_max
    else:
        Sw = bisect(lambda Sw: f_w - get_f_w(mu_w, mu_o, Bo, Bw, Sw, Fw, m1, Fo, m2, Swc, Sor), Sw_min, Sw_max)
        # доп параметры: xtol=0.0001 необходимая точность решения, maxiter=1000 максимальное число итераций
    return Sw


def get_f_w(mu_w, mu_o, Bo, Bw, Sw, Fw, m1, Fo, m2, Swc, Sor):
    """
    Обводненность продукции скважины от водонасыщенности Sw - Функция Баклея-Леверетта
    """
    k_rw = get_k_corey(Fw, m1, Swc, Sor, Sw, type="water")  # ОФП по воде
    k_ro = get_k_corey(Fo, m2, Swc, Sor, Sw, type="oil")  # ОФП по нефти
    try:
        f_w = 100 / (1 + (k_ro * mu_w * Bw) / (k_rw * mu_o * Bo))
    except ZeroDivisionError:
        f_w = 0
    return f_w


def get_k_corey(F, m, Swc, Sor, Sw, type):
    """Относительные фазовые проницаемости по нефти/воде от водонасыщенности Sw (по Кори)"""
    if Sw == 1 and type == "water":
        return 1
    elif Sw == 1 and type == "oil":
        return 0
    else:
        try:
            Sd = (Sw - Swc) / (1 - Sor - Swc)  # Приведенная водонасыщенность пласта
        except ZeroDivisionError:
            Sd = 1
        if type == "water":
            return F * (Sd ** m)
        elif type == 'oil':
            return F * ((1 - Sd) ** m)


def get_mu(Sw, mu_w, mu_o, Fw, m1, Fo, m2, Swc, Sor):
    """
    Эффективная вязкость жидкости (нефть+вода) в пластовых условиях | 1/атм - сП = мПа*с
    """
    k_rw = get_k_corey(Fw, m1, Swc, Sor, Sw, type="water")  # ОФП по воде
    k_ro = get_k_corey(Fo, m2, Swc, Sor, Sw, type="oil")  # ОФП по нефти
    try:
        mu = (mu_w * mu_o) / (k_ro * mu_w + k_rw * mu_o)
    except ZeroDivisionError:
        mu = 0
    return mu


def get_b(Bo, Bw, f_w):
    """
    Эффективный объемный коэффициент расширения жидкости (нефть+вода) | м3/м3
    """
    f_w_r = f_w * Bw / ((100 - f_w) * Bo + f_w * Bw) * 100
    # B = 1 / ((100 - f_w_r) / Bo + f_w_r / Bw) * 100
    B = Bo * (1 - f_w_r / 100) + Bw * f_w_r / 100
    return B