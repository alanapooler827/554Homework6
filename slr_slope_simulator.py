# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# define SLR slope simulator class
class SLR_slope_simulator:

  # initialize the class
  def __init__(
      self,
      beta_0: float,
      beta_1: float,
      x: np.ndarray,
      sigma: float,
      seed: int | None = None
    ):
    self.beta_0 = beta_0
    self.beta_1 = beta_1
    self.sigma = sigma
    self.x = np.array(x)
    self.n = len(self.x)
    self.rng = np.random.default_rng(seed)
    self.slopes = []

  def generate_data(self):
    # calculate y values
    y = self.beta_0 + self.beta_1 * self.x + self.sigma * self.rng.standard_normal(self.n)
    return self.x, y

  def fit_slope(self, x, y):
    # fit linear regression model
    reg = linear_model.LinearRegression()
    reg.fit(x.reshape(-1, 1), y)

    # return slope
    return reg.coef_[0]

  def run_simulations(self, num_simulations):
    # initialize empty list to store slope estimates
    slopes = []

    # calculate slope for specified number of simulations
    for i in range(num_simulations):
      x, y = self.generate_data()
      slope = self.fit_slope(x, y)

      slopes.append(slope)

    self.slopes = np.array(slopes)

  def plot_sampling_distribution(self):
    # check that length of slopes is greater than 0
    if len(self.slopes) == 0:
      print("Error: run_simulations() must be called first")

    # create plot
    plt.hist(self.slopes, bins=30, edgecolor = 'black')
    plt.xlabel("Estimated Slope")
    plt.ylabel("Frequency")
    plt.title("Sampling Distribution of Sample Slope")
    plt.show()

  def find_prob(self, value, sided):
    # check that length of slopes is greater than 0
    if len(self.slopes) == 0:
      print("Error: run_simulations() must be called first")

    if sided == "above":
      prob = np.mean(self.slopes > value)

    elif sided == "below":
        prob = np.mean(self.slopes < value)

    elif sided == 'two-sided':
        prob = np.mean(np.abs(self.slopes) > abs(value))

    # print error if 'sided' does not equal one of the 3 choices
    else:
        print("sided argument must be 'above', 'below', or 'two-sided'")
        return

    return prob