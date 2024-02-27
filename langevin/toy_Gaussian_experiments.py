import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import (
    generate_data,
    SQuantization,
    multishift_quantization,
    variable_multishift_quantization,
)
import pickle, time, os
from math import sqrt
import itertools

########Hyperparameters##########

AUTO_SEED = True

hyper = {
    "b": 20,  # number of clients
    "d": 50,
    "sigma": 1,
    "iid_level": 5,
    "T_tot": 500_000,
    "T_bi": 450_000,
    "T": 200,  # recording interval
    "N": 50,  # Number of recording
    "rep": 2,
    "seed": 42,
}


########Path settings#############
save_path = "/mnt/beegfs/workdir/mahmoud.hegazy"
t = str(time.time()).split(".")[0]
path_figures = save_path + "/toy/" + t  # +"/figures"
path_data = save_path + "/toy/" + t  # +"/data"
os.makedirs(path_figures, exist_ok=True)
os.makedirs(path_data, exist_ok=True)
####################Hyperparameters###########

print("Sampling seed is ", 42)
np.random.seed(42)
b = hyper["b"]  # number of clients
d = hyper["d"]  # dimension
sigma = hyper["sigma"]  # std deviation for data generation
mu = np.zeros(d)  # mean vector for data generation
iid_level = hyper["iid_level"]  # iid level for data generation
n = np.random.randint(low=10, high=200, size=b)  # number of observations per client
L = np.sum(n) / sigma**2  # step size
T_tot = hyper["T_tot"]
T_bi = hyper["T_bi"]
T = hyper["T"]
rep = hyper["rep"]
y, labels = generate_data(b, d, n, "False", iid_level, sigma, mu)
mu_true = np.mean(np.concatenate(y, axis=0), axis=0)
cov_true = sigma**2 / np.sum(n)

###############Data Generation###########

if AUTO_SEED is True:
    hyper["seed"] = int(t) % 1000
    print("Iteration seed is ", hyper["seed"])
    np.random.seed(hyper["seed"])


###########Algorithms###################
def QLSD_star(
    T_total,  # total number of sampling rounds
    T_bi,  # Iteration to start recording from
    step_size,
    init,
    s,  # number of quantization steps (unsigned)
    multishift=True,
    fixed_var=False,
):
    loss = np.zeros(T_total)
    theta = np.zeros((T_total, d))
    bits = np.zeros((T_tot))
    theta[0, :] = init
    x = init
    loss[0] = (1 / (2 * cov_true)) * np.linalg.norm(init - mu_true) ** 2
    qgrad = np.zeros((b, d))
    for t in range(T_total - 1):
        std = np.zeros((b,))
        if multishift is True and fixed_var is False:
            for i in range(b):
                # qgrad[i,:] = SQuantization(s, n[i] * (x - np.mean(y[i],0))/sigma**2)
                qgrad[i, :], std[i], b_grad = multishift_quantization(
                    s, n[i] * (x - np.mean(y[i], 0)) / sigma**2
                )
                bits[t] += b_grad
            if (np.sum(std**2)) * step_size >= 2:
                coef = 0
            else:
                coef = sqrt((2 * step_size) - np.sum(std**2) * step_size**2)
            # print(coef)
            # Update theta on server
            x = (
                x
                - step_size * np.sum(qgrad, axis=0)
                + coef * np.random.normal(0, 1, size=d)
            )
        elif multishift == True and fixed_var == True:
            for i in range(b):
                sigma_quant = sqrt(2 / (step_size * b))  # TODO revise this
                qgrad[i, :], std[i], b_grad = variable_multishift_quantization(
                    sigma_quant, n[i] * (x - np.mean(y[i], 0)) / sigma**2
                )
                bits[t] += b_grad
            x = x - step_size * np.sum(qgrad, axis=0)

        else:
            for i in range(b):
                qgrad[i, :], b_grad = SQuantization(
                    s, n[i] * (x - np.mean(y[i], 0)) / sigma**2
                )
                bits[t] += b_grad
            # Update theta on server
            x = (
                x
                - step_size * np.sum(qgrad, axis=0)
                + np.sqrt(2 * step_size) * np.random.normal(0, 1, size=d)
            )

        theta[t, :] = x
        loss[t] = (1 / (2 * cov_true)) * np.linalg.norm(x - mu_true) ** 2

    return theta, loss, bits


def LSD_star(T_total, T_bi, step_size, init):
    loss = np.zeros(T_total)
    theta = np.zeros((T_total, d))
    theta[0, :] = init
    x = init
    loss[0] = (1 / (2 * cov_true)) * np.linalg.norm(init - mu_true) ** 2
    bits = np.zeros((T_total,))
    grad = np.zeros((b, d))

    for t in range(T_total - 1):
        qb = 0
        for i in range(b):
            grad[i, :] = n[i] * (x - np.mean(y[i], 0)) / sigma**2
            qb += np.sum(np.zeros_like(grad[i, :]) + 1) * grad.itemsize * 8
        bits[t] = qb
        # Update theta on server
        x = (
            x
            - step_size * np.sum(grad, axis=0)
            + np.sqrt(2 * step_size) * np.random.normal(0, 1, size=d)
        )

        theta[t, :] = x
        loss[t] = (1 / (2 * cov_true)) * np.linalg.norm(x - mu_true) ** 2

    return theta, loss, bits


##Experiments are 3 with multishift, 3 without multishift and no quantization

k = [
    "multishift_fixed",
    "qlsdstar_multishift_4",
    "qlsdstar_multishift_8",
    "qlsdstar_multishift_16",
    "qlsdstar_quant_4",
    "qlsdstar_quant_8",
    "qlsdstar_quant_16",
    "lsd",
]
results = {i: np.zeros((T_tot, rep, d)) for i in k}
bits = {i: np.zeros((T_tot, rep)) for i in k}
S = [2**4, 2**8, 2**16]
init = np.zeros(d)


for r in tqdm(range(rep)):
    print("Repetition n°%i started." % r)
    results["multishift_fixed"][:, r, :], _, bits["multishift_fixed"][:, r] = QLSD_star(
        T_tot, 0, 0.01 / L, init, S[0], multishift=True, fixed_var=True
    )
    (
        results["qlsdstar_multishift_4"][:, r, :],
        _,
        bits["qlsdstar_multishift_4"][:, r],
    ) = QLSD_star(T_tot, 0, 0.01 / L, init, S[0], multishift=True, fixed_var=False)
    (
        results["qlsdstar_multishift_8"][:, r, :],
        _,
        bits["qlsdstar_multishift_8"][:, r],
    ) = QLSD_star(T_tot, 0, 0.01 / L, init, S[1], multishift=True, fixed_var=False)
    (
        results["qlsdstar_multishift_16"][:, r, :],
        _,
        bits["qlsdstar_multishift_16"][:, r],
    ) = QLSD_star(T_tot, 0, 0.01 / L, init, S[2], multishift=True, fixed_var=False)
    results["qlsdstar_quant_4"][:, r, :], _, bits["qlsdstar_quant_4"][:, r] = QLSD_star(
        T_tot, 0, 0.01 / L, init, S[0], multishift=False
    )
    results["qlsdstar_quant_8"][:, r, :], _, bits["qlsdstar_quant_8"][:, r] = QLSD_star(
        T_tot, 0, 0.01 / L, init, S[1], multishift=False
    )
    (
        results["qlsdstar_quant_16"][:, r, :],
        _,
        bits["qlsdstar_quant_16"][:, r],
    ) = QLSD_star(T_tot, 0, 0.01 / L, init, S[2], multishift=False)
    results["lsd"][:, r, :], _, bits["lsd"][:, r] = LSD_star(T_tot, 0, 0.01 / L, init)
    print("Repetition n°%i finished." % r)


#############Plotting the Results######################################
# Plot MSE / number of communicated bits
N = hyper["T"]  # how many estimations to consider
# TODO having T=N just for ease of mind
est_true = mu_true
step = int((T_tot - T_bi) / T)
# print(step)
TT = [step * (k + 1) + T_bi for k in range(N)]


################Calculating the MSE###################################
mse = {i: np.zeros(len(TT) - 1) for i in k}

for t in range(len(TT) - 1):
    for i in k:
        # print(T_bi, TT[t])
        est = np.mean(results[i][T_bi : TT[t], :, :], axis=0)
        Bias = np.linalg.norm(np.mean(est, 0) - est_true) ** 2
        Var = np.mean((np.mean(est, 0) - est) ** 2)
        # print(est,Bias,Var)
        mse[i][t] = Bias + Var

theta_mean = {}
bits_mean = {}
for i in k:
    theta_mean[i] = np.mean(results[i], axis=1)
    bits_mean[i] = np.mean(bits[i], axis=1)
# theta_mean = np.mean()
mse["true_mu"] = est_true
mse["theta_mean"] = theta_mean
mse["bits_mean"] = bits_mean
mse["x_axis"] = TT
mse["hyper"] = hyper
# with open(path_data+'/results.pkl', 'wb') as f :
#     pickle.dump(results, f )
with open(path_data + "/mse" + str(hyper["seed"]) + ".pkl", "wb") as f:
    pickle.dump(mse, f)


fig_path = path_data + "/plot.png"
marker = itertools.cycle((",", "+", ".", "o", "*"))

with open(path_data + "/mse" + str(hyper["seed"]) + ".pkl", "rb") as file:
    # Unpickle the dictionary
    data = pickle.load(file)

x_values = data["x_axis"][:-1]
for idx, (label, y_values) in enumerate(data.items()):
    if label in k:
        plt.plot(x_values, y_values, label=label, marker=next(marker))
    print(label)
plt.yscale("log")

# Add labels and a legend
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.legend()
extension = ".png"
# Specify the path to save the plot
save_path = path_data + "/plot" + extension

# Save the plot to the specified path
plt.savefig(save_path)
