import numpy as np
from numpy import log

from scipy.integrate import quad
from scipy.integrate import solve_ivp


def G(a, b1):
    return 1 + b1 * a**3 * (1-a)**20


def G_p(a, b1):
    return a**2*b1*(a - 1)**19*(23*a - 3)


def kappa(a, c1, b1):
    return a ** 3 * b1 * (
        -11639628 * a ** 20 + 245044800 * a ** 19 - 2457254800 * a ** 18 + 15610795200 * a ** 17 -
        70492497075 * a ** 16 + 240614390016 * a ** 15 - 644502830400 * a ** 14 + 1388159942400 * a **
        13 - 2443739898600 * a ** 12 + 3554530761600 * a ** 11 - 4300982221536 * a ** 10 +
        4344426486400 * a ** 9 - 3665609847900 * a ** 8 + 2578011321600 * a ** 7 - 1503839937600 * a **
        6 + 721843170048 * a ** 5 - 281969988300 * a ** 4 + 88461172800 * a ** 3 - 22115293200 * a ** 2
        + 4655851200 * a - 232792560 * log(a)) / 77597520 + a ** 3 * c1 + 1


def w(a, c1, b1):
    return -1 + (G(a, b1) - 1) / (kappa(a, c1, b1) - 1)


def h(a, c1, b1):
    E_sqd = a**(-3)*kappa(a, c1, b1)/kappa(1, c1, b1)
    return E_sqd


def Om_m(a, c1, b1):
    return 1/kappa(a, c1, b1)


def Om_L(a, c1, b1):
    return 1 - Om_m(a, c1, b1)


def h_h(a, c1, b1):
    return -3 * (1 + w(a, c1, b1) * Om_L(a, c1, b1))


def f1(a, x, y, c1, b1, k):
    G_k = G(a, b1)/kappa(a, c1, b1)
    first = (3 - 3/2 * G_k)
    second = (- 3/2 * G_k - a**2 / 3 * k**2 * kappa(1, c1,
              b1)/kappa(a, c1, b1) * G_p(a, b1)/G(a, b1))
    return -first * x/a - second * y/a**2


def ode_system(a, vars, c1, b1, k):
    y, x = vars
    return [x, f1(a, x, y, c1, b1, k)]


def fs8(y, sig, x, a):
    return (sig/y[-1]) * a * x


def integrand(a, c1, b1):
    return -1 / (a**2 * np.sqrt(h(a, c1, b1)))


def dl_func(ai_z, c1, b1):
    integ = quad(integrand, ai_z, 1, args=(c1, b1))
    return integ


def ratio(a, c, c1, b1, d_l1, d_l2):
    return (np.sqrt(h(a, c1, b1) / h(a, c, 0))) * (d_l1 / d_l2)

def chi_squared(params, fs8, a_values, a_data, y_data, cov_inv, om_fdl):
    b1, sig = params
    c1 = 0
    k = 300

    ai, af = 1e-3, 1
    initial_conditions = [ai, af]
    
    ones = np.ones(len(a_values))
    if not (kappa(a_values,c1,b1)>=ones).all():
        return 1e+6

    sol = solve_ivp(ode_system, [ai, af], initial_conditions, args=(c1, b1, k), t_eval=a_values)
    y_g_s, y_g_prime_s = sol.y

    y_pred = fs8(y_g_s, sig, y_g_prime_s, a_values)

    indices = np.searchsorted(a_values, a_data, side="left")
    preds = y_pred[indices]

    ratios = []
    for i, ai_z in enumerate(a_data):
        dl1, er1 = dl_func(ai_z, c1, b1)
        c = (1/om_fdl[i]) - 1
        dl2, er2 = dl_func(ai_z, c, 0)
        ratios.append(ratio(ai_z, c, c1, b1 , dl1, dl2))

    ratios = np.array(ratios)

    diffs = (y_data - ratios * preds)[::-1]
    chi2 = np.dot(diffs.T, np.dot(cov_inv, diffs))

    return chi2
