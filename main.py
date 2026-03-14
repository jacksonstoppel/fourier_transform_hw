import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fft import fft
from scipy.integrate import solve_ivp

# define math functions for problems
def prob1_analytic(k):
    return np.sqrt(np.pi / 50.0) * np.exp(-(np.pi**2) * k**2 / 50.0) * np.exp(-1j * np.pi * k)

def func_prob1(x):
    return np.exp(-50.0 * (x - 0.5)**2)


    

# create DFT and inverse DFT
def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    #sum over all k and x values
    for k in range(N):
        s = 0.0 + 0.0j
        for n in range(N):
            s += x[n] * np.exp(-2j * np.pi * k * n / N)
        X[k] = s / N
    return X

# inverse DFT
def inverse_DFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        s = 0.0 + 0.0j
        for k in range(N):
            s += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] = s
    return x


# Generate the correct k values
def signed_k_vals(N):
    k = np.arange(N)
    return np.where(k < N // 2, k, k - N)

Ns = [32, 64, 128]

plt.figure(figsize=(8, 5))

# create and plot the data for problem 1
k_cont = np.linspace(-10, 10, 1000)
plt.plot(k_cont, np.abs(prob1_analytic(k_cont)), label='Analytical FT', linewidth=2)

for N in Ns:
    x_grid = np.arange(N) / N
    x_samples = func_prob1(x_grid)

    X = DFT(x_samples)
    k_signed = signed_k_vals(N)

    order = np.argsort(k_signed)
    plt.plot(k_signed[order], np.abs(X)[order], 'o-', label=f'Manual DFT, N={N}', markersize=4)

plt.xlabel('k')
plt.ylabel(r'$|\hat f(k)|$ or $|X_k|$')
plt.title('Analytical FT vs manual DFT on [0,1]')
plt.legend()
plt.grid(True)
plt.show()

for N in Ns:
    x_grid = np.arange(N) / N
    x_samples = func_prob1(x_grid)

    X = DFT(x_samples)
    x_recon = inverse_DFT(X)

    plt.figure(figsize=(8, 4))
    plt.plot(x_grid, x_samples, 'k-', label='Original samples')
    plt.plot(x_grid, x_recon.real, 'ro', label='Reconstructed from Inverse DFT', markersize=4)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Reconstruction check, N={N}')
    plt.legend()
    plt.grid(True)
    plt.show()

    max_err = np.max(np.abs(x_samples - x_recon.real))
    print(f'N = {N}, max reconstruction error = {max_err:.3e}')

N_fft_demo = 128
x_grid = np.arange(N_fft_demo) / N_fft_demo
x_samples = func_prob1(x_grid)

X_manual = DFT(x_samples)
X_fft = fft(x_samples) / N_fft_demo

k_signed = signed_k_vals(N_fft_demo)
order = np.argsort(k_signed)

plt.figure(figsize=(8, 5))
plt.plot(k_signed[order], np.abs(X_manual)[order], 'o-', label='DFT')
plt.plot(k_signed[order], np.abs(X_fft)[order], '--', label='FFT / N')
plt.xlabel('k')
plt.ylabel('Magnitude')
plt.title('DFT vs FFT')
plt.legend()
plt.grid(True)
plt.show()


sizes = [16, 32, 64, 128, 256, 512]
t_dft = []
t_fft = []

for N in sizes:
    x_grid = np.arange(N) / N
    x_samples = func_prob1(x_grid)

    # time manual DFT
    start = time.perf_counter()
    DFT(x_samples)
    end = time.perf_counter()
    t_dft.append(end - start)

    # time FFT
    start = time.perf_counter()
    fft(x_samples) / N
    end = time.perf_counter()
    t_fft.append(end - start)

plt.figure(figsize=(8, 5))
plt.loglog(sizes, t_dft, 'o-', label='DFT')
plt.loglog(sizes, t_fft, 's-', label='FFT')
plt.xlabel('N')
plt.ylabel('Runtime (s)')
plt.title('Time complexity: manual DFT vs FFT')
plt.legend()
plt.grid(True, which='both')
plt.show()


# start problem 2

alpha = 0.01
N = 64
Tfinal = 0.5

x = np.arange(N) / N
u0 = func_prob1(x)

# transform to k-space
U0 = DFT(u0)
k_vals = signed_k_vals(N)

# function for the right side of the equation
def rhs(t, U):
    return -alpha * (2 * np.pi * k_vals)**2 * U


t_eval = np.linspace(0, Tfinal, 150)


sol = solve_ivp(
    rhs,
    t_span=(0.0, Tfinal),
    y0=U0,
    t_eval=t_eval,
    method='RK45'
)


u_xt = np.zeros((len(t_eval), N))
for j in range(len(t_eval)):
    u_xt[j, :] = inverse_DFT(sol.y[:, j]).real


Xgrid, Tgrid = np.meshgrid(x, t_eval)

plt.figure(figsize=(8, 5))
plt.pcolormesh(Xgrid, Tgrid, u_xt, shading='auto')
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar(label='u(x,t)')
plt.title('Heat equation solved in k-space')
plt.show()


plt.figure(figsize=(8, 5))
for idx in [0, len(t_eval)//4, len(t_eval)//2, -1]:
    plt.plot(x, u_xt[idx, :], label=f't = {t_eval[idx]:.3f}')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()
