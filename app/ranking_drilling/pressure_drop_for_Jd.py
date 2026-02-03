from math import log, fabs, sqrt, exp, sin, cos,  pi
from mpmath import invertlaplace, nsum, inf, tanh
from scipy.special import k0  # Модифицированная функция Бесселя второго рода
from scipy.integrate import quad


def calc_xd(cfd):
    """
    Аппроксимация функции xd(cfd) полиномом для расчета падения давления по трещине с конечной проводимостью в модели трилинейного потока
    SPE 26424
    Уравнение получено для диапазона 0,5 ~ CfD ~ 10,000
    xD=0,732 может быть использовано для трещин бесконечной проводимости
    """
    # коэффициенты полиномиальной функции
    a0, a1, a2, a3, a4 = 0.759919, 0.465301, 0.562754, 0.363093, 0.0298881
    b0, b1, b2, b3, b4 = 1, 0.99477, 0.896679, 0.430707, 0.0467339
    var = log(cfd)
    xd_cfd = ((a0 + a1 * var + a2 * var ** 2 + a3 * var ** 3 + a4 * var ** 4) /
              (b0 + b1 * var + b2 * var ** 2 + b3 * var ** 3 + b4 * var ** 4))
    return xd_cfd


def sum_func(func):
    """Сумма членов уравнений"""
    summa = nsum(func, [1, inf])  # , tol=1e-20)
    return summa


def pd_b3_integrate(u, xd, xwd, xed, yd, ywd, k, sgn1, sgn2):
    """Интегралы в расчете pd_b3"""
    func_K_0 = lambda alpha: k0(sqrt((xd + sgn1 * xwd + sgn2 * 2 * k * xed - alpha) ** 2 + (yd - ywd) ** 2) * u)
    pd_b3_integral, _ = quad(func_K_0, -1, 1, epsabs=1e-6, epsrel=1e-6)
    return pd_b3_integral


def pd_b3(S, xd, xwd, xed, yd, ywd):
    """Вычисляет вклад третьего члена в разложение давления"""
    u = sqrt(S)
    part_1 = pd_b3_integrate(u, xd, xwd, xed, yd, ywd, k=0, sgn1=1, sgn2=1)

    # Возможные значения sgn1 и sgn2
    signs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    part_2 = sum_func(lambda k: sum(pd_b3_integrate(u, xd, xwd, xed, yd, ywd, k, sgn1, sgn2) for sgn1, sgn2 in signs))
    part_3 = pi * exp(-u * fabs(yd - ywd)) / (xed * S * u)
    pd_b3 = 1 / (2 * S) * (part_1 + part_2) - part_3
    return pd_b3


def pd_b2_k(k, S, xd, xwd, xed, yd, ywd, yed, yd1, yd2):
    """Множители второго члена"""
    ek = sqrt(S + k ** 2 * pi ** 2 / xed ** 2)
    summa_exp = sum_func(lambda m: exp(-2 * m * ek * yed))
    part_1 = sin(k * pi / xed) * cos(k * pi * xd / xed) * cos(k * pi * xwd / xed) / (k * ek)
    part_2 = (exp(-ek * (yd + ywd)) + exp(-ek * (yed + yd1))
              + exp(-ek * (yed + yd2))) * (1 + summa_exp) + exp(-ek * fabs(yd - ywd)) * summa_exp
    pd_b2_k = part_1 * part_2
    return pd_b2_k


def pd_b2(S, xd, xwd, xed, yd, ywd, yed, yd1, yd2):
    """
    Вычисляет вклад второго члена в разложение давления.
    Учитывает периодическое распределение давление вдоль осей x и y.
    """
    Sum = sum_func(lambda k: pd_b2_k(k, S, xd, xwd, xed, yd, ywd, yed, yd1, yd2))
    pd_b2 = Sum * 2 / S
    return pd_b2


def pd_b1(S, yd, ywd, yed, xed, yd1, yd2):
    """
    Вычисляет вклад первого члена в разложение давления.
    Вклад давления, связанный с границами области по оси y, где присутствует экспоненциальное затухание давления.
    Имеет значительный вклад, если расстояния между скважинами и границами области по оси y малы.
    """
    u = sqrt(S)
    summa_exp = sum_func(lambda m: exp(-2 * m * u * yed))
    pd_b1 = (pi / (xed * S * u) *
             (exp(-u * (yd + ywd)) + exp(-u * (yed + yd2)) + exp(-u * fabs(yd - ywd)) + exp(-u * (yed + yd1)))
             * (1 + summa_exp))
    return pd_b1


def integral_k0(x):
    """Вычисляет интеграл модифицированной функции Бесселя второго рода K_0(x) от 0 до x"""
    if x <= 0:
        return 0
    else:
        func_K_0 = lambda f: k0(f)
        result, _ = quad(func_K_0, 0, x, epsabs=1e-6, epsrel=1e-6)  # Вычисление определенного интеграла от 0 до x
        return result


def pd_inf(S, xd, xwd):
    """
    Решение Ozkan-Raghavan в пространствее Лапласа
    для скважины с равномерным потоком/вертикальной трещиной бесконечной проводимостью в бесконечном однородном пласте
    """
    if fabs(xd - xwd) < 1:
        lim_1 = sqrt(S) * (1 + (xd - xwd))
        lim_2 = sqrt(S) * (1 - (xd - xwd))
        sign = 1
    else:
        lim_1 = sqrt(S) * (1 + fabs(xd - xwd))
        lim_2 = sqrt(S) * (fabs(xd - xwd) - 1)
        sign = -1
    intgr = integral_k0(lim_1) + sign * integral_k0(lim_2)
    pd_inf = 1 / (2 * S * sqrt(S)) * intgr
    return pd_inf


def pd_OR(S, xd, xwd, xed, yd, ywd, yed):
    """
    Решение Ozkan и Raghavan в пространстве Лапласа для вертикальной трещины в ограниченной прямоугольной области
    (учет псевдорадиального течения)
    pd_b - вклад границ пласта (члены учета ограниченного пласта)
    pd_inf - решение в бесконечном пласте
    """
    yd1 = yed - fabs(yd - ywd)
    yd2 = yed - (yd + ywd)
    p_wD = (pd_inf(S, xd, xwd) +
            pd_b1(S, yd, ywd, yed, xed, yd1, yd2) +
            pd_b2(S, xd, xwd, xed, yd, ywd, yed, yd1, yd2) +
            pd_b3(S, xd, xwd, xed, yd, ywd))
    return p_wD


def pd_LB(S, cfd, skin_f, Cd_fi, Cd, eta):
    """
    Решение Lee and Brockenbrough трилинейного потока для скважины с конечной/бесконечной проводимостью в бесконечном
    однородном пласте в ранний период (не работает для псевдорадильного течения)
    Parameters
    ----------
    S - параметр Лапласа
    cfd = 10000000 - безразмерный коэффициент проводимости трещин ГРП
    skin_f - скин
    eta - Dimensionless fracture diffusivity - безразмерная диффузия трещины
    Cd_fi - Dimensionless phase redistribution pressure parameter
    Cd(в РБ s_wfd) - Dimensionless wellbore storage - безразмерная ёмкость ствола скважины
    """
    # Параметры для упрощения уравнения
    beta = pi / cfd
    alpha = (S + S ** 0.5) ** 0.5
    psi = ((S / eta) + 2 * alpha / (cfd * (1 + 2 / pi * skin_f * alpha))) ** 0.5
    # Учет роста давления за счет перераспределения фаз внутри ствола скважины
    pd_fi = Cd_fi / (cfd * S ** 2 + S)  # Dimensionless phase redistribution pressure
    # Расчет давления
    p_wD = beta * (1 + S ** 2 * Cd * pd_fi) / S / (psi * tanh(psi) + (beta * S * Cd))
    return p_wD


def PdFracWBSSf(S, cfd, skin_f, Cd, Cd_fi, lf, mult=100000):
    """
    Безразмерное падение давления Pd, вызванное трещиной с конечной проводимостью
    Учитывает ёмкость трещины Frac, скин-эффект Sf, влияние ствола скважины WBS

    Parameters
    ----------
    S - переменная Лапласа
    cfd - безразмерный коэффициент проводимости трещин
    skin_f - скин
    Cd (в РБ s_wfd) - Dimensionless wellbore storage - безразмерная ёмкость ствола скважины (принят 0)
    Cd_fi - dimensionless phase redistribution pressure parameter
    lf - полудлина трещины
    """
    eta_f = 2 * cfd * lf / 0.005  # безразмерная диффузия трещины (порядка 1*10^10)
    eta_inf = 2 * mult * cfd * lf / 0.005

    # Конечная проводимость (finite)
    p_wD_f = pd_LB(S, cfd, skin_f, Cd_fi, Cd, eta_f)
    # Бесконечная проводимость - вычитание для исключения двойного учета "идеальной трещины" в LB и OR
    p_wD_inf = pd_LB(S, mult * cfd, 0, 0, 0, eta_inf)

    return p_wD_f - p_wD_inf


def get_dimensionless_delta_pressure(t, cfd, lf, k_h, c_t, porosity, mu, xe, ye, xw, yw, mode, skin_f=0, Cd=0, Cd_fi=1):
    """
    Безразмерный перепад давления  Pd в пласте с прямоугольной геометрией и без течения флюида через границы
    Parameters
    ----------
    t - время работы скважины, часы
    cfd - безразмерный коэффициент проводимости трещин ГРП, в случае скважины без ГРП - 10000000
    lf - полу-длина трещины
    k_h - проницаемость
    c_t - общая сжимаемость
    porosity - пористость
    mu - вязкость флюида
    xe - размер прямоугольной области дренирования по x
    ye - размер прямоугольной области дренирования по y
    xw - координата скважины x
    yw - координата скважины y
    mode - тип расчета
    skin_f - скин фактор
    Cd (в РБ s_wfd) - Dimensionless wellbore storage - безразмерная ёмкость ствола скважины
            Cd = 0.17 * С / (Phi  * c_t  * h  * lf ** 2)
            С (в РБ wbs) - Wellbore Storage Constant (bbl/psi или м3/МПа) - константа ёмкости ствола скважины
    Cd_fi - dimensionless phase redistribution pressure parameter
    Returns
    -------
    Pwd - безразмерный перепад давления
    """
    # безразмерное время (0.00036 при размерностях м, сек, атм)
    td = 0.00036 * k_h * t / (porosity * mu * c_t * lf ** 2)

    # безразмерные координаты скважины
    xwd, ywd = xw / lf, yw / lf

    # безразмерные координаты границ дренажного прямоугольника xe и ye
    xed, yed = xe / lf, ye / lf

    # безразмерные координаты точки, в которой рассчитывается давление
    xd = calc_xd(cfd) + xwd  # dimensionless distance along the fracture - безразмерное расстояние вдоль трещины
    yd = ywd  # dimensionless distance perpendicular to fracture - безразмерное расстояние перпендикулярно трещине

    pw_rect_stehf = 0
    N = 8  # количество членов суммы, используемых для приближения численного преобразования Лапласа в методе Штефеста
    # S = i * log(2) / td - переменная интегрирования пространства Лапласа, используемая для расчета давления во времени
    # Использование алгоритма Штефеста - Численная инверсия для каждого момента времени
    if mode == 1:  # well flowing pressure
        P_laplace = lambda S: (PdFracWBSSf(S, cfd, skin_f, Cd, Cd_fi, lf)
                               + pd_OR(S, xd, xwd, xed, yd, ywd, yed))
        # Используем invertlaplace для вычисления давления во времени
        pw_rect_stehf = invertlaplace(P_laplace, td, method='stehfest', degree=N)

    elif mode == 2:  # well flow rate
        P_laplace = lambda S: 1 / S ** 2 / (PdFracWBSSf(S, cfd, skin_f, Cd, Cd_fi, lf)
                                            + pd_OR(S, xd, xwd, xed, yd, ywd, yed))
        pw_rect_stehf = invertlaplace(P_laplace, td, method='stehfest', degree=N)
    return pw_rect_stehf
