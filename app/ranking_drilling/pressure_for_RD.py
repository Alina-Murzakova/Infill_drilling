from math import log, fabs, sqrt, exp, sin, cos, tanh, pi
from mpmath import invertlaplace, nsum, inf
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


def sum_func(func):
    """Сумма членов уравнений"""
    summa = nsum(func, [1, inf])  # , tol=1e-20)
    return summa


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
    Безразмерное давление на поверхности трещины
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


def pd_lapl_rect(S, xd, xwd, xed, yd, ywd, yed):  # Rectangle
    """
    Метод Лапласа для расчета безразмерного давления в прямоугольной области - для решения уравнения фильтрации
    pd_b - члены учета ограниченного пласта
    pd_inf - решение в бесконечном пласте
    """
    yd1 = yed - fabs(yd - ywd)
    yd2 = yed - (yd + ywd)
    pd_lapl_rect = (pd_inf(S, xd, xwd) +
                    pd_b1(S, yd, ywd, yed, xed, yd1, yd2) +
                    pd_b2(S, xd, xwd, xed, yd, ywd, yed, yd1, yd2) +
                    pd_b3(S, xd, xwd, xed, yd, ywd))
    return pd_lapl_rect


def p_wbs(S, cfd, skin_f, c_fi, s_wfd, eta):
    """
    Решение Lee and Brockenbrough трилинейного потока для скважины с конечной/бесконечной проводимостью в бесконечном
    однородном пласте в ранний период
    Влияние ствола скважины и скин-эффектов
    Безразмерное давление на стенке скважины в пространстве Лапласа в ранний период
    Parameters
    ----------
    S - параметр Лапласа
    cfd = 10000000 - безразмерный коэффициент проводимости трещин ГРП
    skin_f - скин
    eta - Dimensionless fracture diffusivity - безразмерная диффузия трещины
    """
    # Параметры для упрощения уравнения
    beta = pi / cfd
    u = (S + S ** 0.5) ** 0.5
    psi = (1 * (S / eta) + 2 * u / cfd / (1 + 2 / pi * skin_f * u)) ** 0.5
    pfi = c_fi / (cfd * S ** 2 + S)  # Dimensionless Phase Segregation Pressure Rise
    # Расчет давления
    p_wbs = beta * (1 + S ** 2 * s_wfd * pfi) / S / (psi * tanh(psi) + (beta * S * s_wfd))
    return p_wbs


def PdFracWBSSf(S, cfd, skin_f, s_wfd, c_fi, lf, mult=100000):
    """
    Функция, моделирующая поведение трещины - моделирует влияние трещины на распределение давления
    Parameters
    ----------
    S - переменная Лапласа
    cfd - безразмерный коэффициент проводимости трещин
    skin_f - скин
    lf - полудлина трещины

    Returns
    -------
    PdFracWBSSf - разница в поведении давления между трещиной с бесконечной проводимостью
                                                    и трещиной с конечной проводимостью
    """
    eta_f = 2 * cfd * lf / 0.005  # hydraulic diffusivity ratio for the fracture (порядка 1*10^10)
    eta_inf = 2 * mult * cfd * lf / 0.005

    # Конечная проводимость
    p_wD_f = p_wbs(S, cfd, skin_f, c_fi, s_wfd, eta_f)
    # Бесконечная проводимость
    p_wD_inf = p_wbs(S, mult * cfd, 0, 0, 0, eta_inf)

    return p_wD_f - p_wD_inf


def Pwd(t, cfd, lf, k_h, c_t, Phi, mu, xe, ye, xw, yw, mode, skin_f=0, s_wfd=0, c_fi=1):
    """
    Безразмерный перепад давления  Pw в пласте с прямоугольной геометрией и без течения флюида через границы
    Parameters
    ----------
    t - время работы скважины, часы
    cfd - безразмерный коэффициент проводимости трещин ГРП, в случае скважины без ГРП - 10000000
    lf - полудлина трещины
    k_h - проницаемость
    c_t - общая сжимаемость
    Phi - пористость
    mu - вязкость флюида
    xe - размер прямоугольной области дренирования по x
    ye - размер прямоугольной области дренирования по y
    xw - координата скважины x
    yw - координата скважины y
    mode - тип расчета
    skin_f - скин фактор
    s_wfd - безразмерный параметр интенсивность притока флюида зависит от модели двойной пористости
            s_wfd = 0.17 * wbs / Phi / c_t / h / lf ** 2 # Cd - Dimensionless wellbore storage
            wbs - Wellbore Storage Constant (bbl/psi) - Объем хранилища скважины - отражает объем флюида,
            который может храниться в стволе скважины
    c_fi - ?
    Returns
    -------
    Pwd - безразмерный перепад давления
    """
    # безразмерное время (0.00036 при размерностях м, сек, атм)
    td = 0.00036 * k_h * t / (Phi * mu * c_t * lf ** 2)

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
        P_laplace = lambda S: (PdFracWBSSf(S, cfd, skin_f, s_wfd, c_fi, lf)
                               + pd_lapl_rect(S, xd, xwd, xed, yd, ywd, yed))
        # Используем invertlaplace для вычисления давления во времени
        pw_rect_stehf = invertlaplace(P_laplace, td, method='stehfest', degree=N)

    elif mode == 2:  # well flow rate
        P_laplace = lambda S: 1 / S ** 2 / (PdFracWBSSf(S, cfd, skin_f, s_wfd, c_fi, lf)
                                            + pd_lapl_rect(S, xd, xwd, xed, yd, ywd, yed))
        pw_rect_stehf = invertlaplace(P_laplace, td, method='stehfest', degree=N)
    return pw_rect_stehf
