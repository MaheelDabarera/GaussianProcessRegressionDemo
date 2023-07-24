# Gaussian Process Regression

This code implements Gaussian process regression to estimate the function

def fpure(v):
    return np.cos(v)**2

# Define the function to estimate
noise_std = 0.2
N = 260
x_init = np.linspace(0, 4 * np.pi, N)
f = fpure(x_init) + noise_std * np.random.randn(x_init.shape)

# Randomize the data
n = len(f)
A = np.array([x_init, f]).T
A_rand = A[np.random.permutation(n), :]
x_rand = A_rand[:, 0]
y_rand = A_rand[:, 1]

# Split the data into training and testing sets
y = y_rand[:0.8 * len(y_rand)]
ys = y_rand[0.8 * len(y_rand):]
x = x_rand[:0.8 * len(x_rand)]
xs = x_rand[0.8 * len(x_rand):]

# Transform the data
y_init = y

# y, lambda = boxcox(y)
ys_init = ys
# ys = boxcox(lambda, ys)

# Normalize the outputs
ynorm = (y - np.mean(y)) / np.std(y)
ysnorm = (ys - np.mean(y)) / np.std(y)

# Normalize the inputs
xnorm = (x - np.mean(x)) / np.std(x)
xsnorm = (xs - np.mean(x)) / np.std(x)

# Specify the Gaussian Process
mean_func = None
cov_func = {'linear': 1.0}
lik_func = 'gauss'
inf = 'exact'
sn = 0.1
hyp0 = {'cov': np.zeros((2, 1)), 'lik': np.log(sn)}
Kx = feval(cov_func, hyp0['cov'], xnorm)
Kxs = feval(cov_func, hyp0['cov'], xnorm, xsnorm)

# Fit the model
def minimize(hyp0, func, maxiter, inf, mean_func, cov_func, lik_func, x, y):

  def callback(hyp):
        print(hyp['cov'])

  best_hyp, best_val, _ = scipy.optimize.minimize(
        func, hyp0, args=(x, y), method='L-BFGS-B', callback=callback, options={'maxiter': maxiter})
    return best_hyp

# Make predictions
mu, s2 = gp_predict(hyp, inf, mean_func, cov_func, lik_func, xnorm, ynorm, xsnorm)

# Get the proper scalings for comparison
mu_unscaled = np.std(y) * mu + np.mean(y)

# Plot the predictions
plt.plot(xs, mu_unscaled, 'o')
plt.show()

# Plot the true function
plt.plot(x_init, f)
plt.show()

# Calculate the mean squared prediction error
mspe = np.mean((mu_unscaled - ys)**2)
print('The mean squared prediction error is %f' % mspe)


To use this code, you will need to have Python 3 installed. Once you have Python 3 installed, you can run the code by following these steps:

Clone the repository to your computer.
Open a terminal window and navigate to the directory where you cloned the repository.

Run the following command:

"
python gp.py
"

This will run the Gaussian process regression code and print the mean squared prediction error. Refer to the .py file for the full code.
