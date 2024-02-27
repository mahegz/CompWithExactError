import numpy as np
import scipy as sp


# log (a choose b)
def logbinom(a, b):
    return (
        -sp.special.loggamma(b + 1)
        + sp.special.loggamma(a + 1)
        - sp.special.loggamma(a - b + 1)
    )


# Pdf of Irwin-Hall distribution
# r:    specify length of support (support is [-r/2, r/2])
# sd:   specify standard deviation
# log:  compute log of pdf instead
# diff: compute derivative of pdf instead
def irwin_hall(
    n, r=None, sd=None, d2=None, log=False, log_in=True, diff=False, return_supp=False
):
    a = 1.0
    b = 0.0

    if r is not None:
        a = r / n
        b = -n / 2
    elif sd is not None:
        a = sd / np.sqrt(n / 12)
        b = -n / 2
    elif d2 is not None:
        cd2 = -(
            sum(
                (-1) ** k * sp.special.binom(n, k) * (n / 2 - k) ** (n - 3)
                for k in range(n // 2 + 1)
            )
            / sp.special.gamma(n - 2)
        )
        a = np.sqrt(cd2 / d2)
        b = -n / 2

    def f(x):
        if isinstance(x, np.random.Generator):
            return (sum(x.random() for _ in range(n)) + b) * a

        y = x / a - b
        if log or log_in:
            if y <= 0 or y >= n:
                if log:
                    return -np.inf
                else:
                    return 0.0

            ret = 0.0
            ub = int(y - 1e-9) + 1
            if diff:
                if y >= n / 2:
                    if log:
                        return -np.inf
                    else:
                        return 0.0

                ret = (
                    sp.special.logsumexp(
                        [logbinom(n, k) + (n - 2) * np.log(y - k) for k in range(ub)],
                        b=[(-1) ** k for k in range(ub)],
                    )
                    - sp.special.loggamma(n - 1)
                    - np.log(a) * 2
                )
            else:
                ret = (
                    sp.special.logsumexp(
                        [logbinom(n, k) + (n - 1) * np.log(y - k) for k in range(ub)],
                        b=[(-1) ** k for k in range(ub)],
                    )
                    - sp.special.loggamma(n)
                    - np.log(a)
                )
            if log:
                return ret
            else:
                if np.isnan(ret):
                    return 0.0
                return np.exp(ret)

        else:
            if y < 0 or y > n:
                return 0.0
            if diff:
                return (
                    sum(
                        (-1) ** k * sp.special.binom(n, k) * (y - k) ** (n - 2)
                        for k in range(int(y) + 1)
                    )
                    / sp.special.gamma(n - 1)
                    / a ** 2
                )
            else:
                return (
                    sum(
                        (-1) ** k * sp.special.binom(n, k) * (y - k) ** (n - 1)
                        for k in range(int(y) + 1)
                    )
                    / sp.special.gamma(n)
                    / a
                )

    if return_supp:
        return (f, (b * a, (n + b) * a))
    else:
        return f


# Inverse pdf of Irwin-Hall with support [-1/2,1/2]
def inv_irwin_hall(n, r=None, sd=None):
    f, supp = irwin_hall(n, r=r, sd=sd, return_supp=True)
    return lambda y: sp.optimize.bisect(
        lambda x: f(x) - y, (supp[0] + supp[1]) * 0.5, supp[1]
    )


# Gaussian pdf
def gauss(x, mu=0, sd=1):
    return sp.stats.norm.pdf(x, mu, sd)


# Log Gaussian pdf
def loggauss(x):
    return -0.5 * np.log(2 * np.pi) - 0.5 * x ** 2


# Inverse of Gaussian pdf
def invgauss(y):
    y = np.log(y)
    return np.sqrt(-2 * (y + 0.5 * np.log(2 * np.pi)))


# Derivative of Gaussian pdf
def diffgauss(x):
    return -x * gauss(x)


# Log derivative of Gaussian pdf
def logdiffgauss(x):
    # print(x)
    if x >= 0.0:
        return 0.0
    return -0.5 * np.log(2 * np.pi) - 0.5 * x ** 2 + np.log(-x)


# Min ratio of derivative of Gaussian and Irwin-Hall pdfs
def minratio(n, sd_scale=1.0, diff=True):
    if diff and n <= 2:
        return 0.0

    f, supp = irwin_hall(n, sd=sd_scale, log=True, diff=diff, return_supp=True)
    g = logdiffgauss if diff else loggauss

    res = sp.optimize.minimize_scalar(
        lambda x: max(g(x) - f(x), -10), bracket=(supp[0] + 1e-7, -1e-7)
    )

    return np.exp(res.fun)


# The bound on relative mixture entropy
def mixture_ent_bd(n):
    f, supp = irwin_hall(n, sd=1, return_supp=True)
    lam = minratio(n)
    L = supp[1] - supp[0]
    f0 = f(0)
    g0 = gauss(0)
    return (1 - lam) * (
        L * f0 + np.log2((np.e * L * (g0 - lam * f0)) / (2 * (1 - lam)))
    )


# The bound on the expected length for aggregated Gaussian
def len_bd(n):
    hM = mixture_ent_bd(n)
    sqrt3n = np.sqrt(n * 3)
    f, supp = irwin_hall(n, sd=1.0, return_supp=True)
    # EabsQ = sp.integrate.quad(lambda x: gauss(x)*abs(x), -10.0, 10.0)[0]
    # EabsP = sp.integrate.quad(lambda x: f(x)*abs(x), supp[0], supp[1])[0]

    def rf(t, sd):
        return (
            hM
            + np.log2(t / (2 * sd * sqrt3n))
            + 1
            + (6 * sd * sqrt3n * np.log2(np.e)) / t
        )

    return rf


# The bound on the expected length for individual Gaussian
def len_bd_gauss():
    def tf(x):
        l = 2 * invgauss(x)
        return l * np.log2(l)

    hL = sp.integrate.quad(tf, 1e-11, gauss(0))[0]
    EabsQ = sp.integrate.quad(lambda x: gauss(x) * abs(x), -10.0, 10.0)[0]

    # print(hL)
    def rf(n, t, sd):
        # c = sd / np.sqrt(n)
        c = sd * np.sqrt(n)
        return -hL + np.log2(t / c) + 8 * (np.log2(np.e) * c / t) * EabsQ + 1

    return rf


# The bound on the expected length for Irwin-Hall
def len_bd_dif(n, t, sd):
    # d = 2*sd*np.sqrt(3.0/n)
    d = 2 * sd * np.sqrt(3.0 * n)
    return np.log2(t / d + 3) + 1


# Decompose Unif(-1/2,1/2) into Irwin-Hall
def decompose_unif(n, rng):
    a = 1.0
    b = 0.0
    f = irwin_hall(n, r=1.0)
    invf = inv_irwin_hall(n, r=1.0)
    f0 = f(0.0)
    while True:
        u = rng.random() - 0.5
        v = rng.random()
        if v <= f(u) / f0:
            return (a, b)
        s = invf(v * f0)
        b += a * np.sign(u) * (s + 0.5) / 2
        a *= 0.5 - s


# Decompose Gaussian into Irwin-Hall
def decompose(n, rng):
    f, supp = irwin_hall(n, sd=1, return_supp=True)
    lam = minratio(n)
    L = supp[1] - supp[0]
    x = rng.normal()
    v = gauss(x) * rng.random()
    if v > gauss(x) - lam * f(x):
        return (1.0, 0.0)

    s = sp.optimize.bisect(lambda x2: gauss(x2) - lam * f(x2) - v, 0.0, 12.0)
    a, b = decompose_unif(n, rng)
    return (2 * a * s / L, 2 * b * s)


# Encoding function
def encode(x, n, i, sd, rng, a=None, b=None, s=None):
    if a is None or b is None:
        a, b = decompose(n, rng)
    if s is None:
        s = [rng.random() - 0.5 for _ in range(n)]
    d = 2 * sd * np.sqrt(3 * n)
    return round(x / (a * d) + s[i])


# Decoding function
def decode(m, n, sd, rng, a=None, b=None, s=None):
    if a is None or b is None:
        a, b = decompose(n, rng)
    if s is None:
        s = [rng.random() - 0.5 for _ in range(n)]
    d = 2 * sd * np.sqrt(3 * n)
    return (a * d) / n * (sum(m) - sum(s)) + b * sd
