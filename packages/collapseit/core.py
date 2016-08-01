import numpy as np
import scipy as sp
import scipy.optimize as opt
import warnings
import logging

# Configure logging for this module
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def master_curve(hypothesis, parameters, data, x_eval):
  """
  Parameters
  ----------
  hypothesis : Scaling hypothesis.
  parameters : Parameters for the scaling hypothesis.
  data
  x_eval     : Scaled values at which to evaluate the master curve.

  Returns
  -------
  Y
  dY
  """
  # Scale the data and the errors
  scaled_data = [(data_L['L'],
                  hypothesis.scale(parameters,
                                   data_L['L'],
                                   (data_L['data'][:,0],
                                    data_L['data'][:,1],
                                    data_L['data'][:,2])))
                 for data_L in data]

  # Loop over all the points at which to evaluate the master curve.
  Y_v = []; dY2_v = []
  for v in x_eval:

    # Select the neighbouring points at which to evaluate for the regression.
    v_data = []
    for L, (x,y,dy) in scaled_data:
      # Pick the data point just above and just below the target point
      gt = (x - v) > 0.0
      lt = (x - v) < 0.0
      x_gt = x[gt]; x_lt = x[lt]
      if x_gt.size == 0 or x_lt.size == 0:
        continue
      upper = np.argmin(x_gt - v)
      lower = np.argmax(x_lt - v)
      upper = x_gt[upper], y[gt][upper], dy[gt][upper]
      lower = x_lt[lower], y[lt][lower], dy[lt][lower]

      v_data.extend([upper, lower])

    if len(v_data) == 0:
      Y_v.append(np.nan)
      dY2_v.append(np.nan)
      continue

    # Perform a linear regression over the points chosen
    v_data = np.array(v_data)
    x,y,dy = v_data[:,0], v_data[:,1], v_data[:,2]
    w = 1. / np.square(dy)
    K = np.sum(w)
    K_x = np.dot(w,x)
    K_y = np.dot(w,y)
    K_xx = np.dot(w, x*x)
    K_xy = np.dot(w, x*y)
    
    Delta = K * K_xx - K_x * K_x
    A = (K_y * K_xx - K_x * K_xy) / Delta
    B = (K * K_xy - K_x * K_y) / Delta
    Y_v.append(A + B * v)
    dY2_v.append((K_xx - 2. * v * K_x + v*v * K) / Delta)
  return np.array(Y_v), np.sqrt(np.array(dY2_v))

def cost(hypothesis, parameters, data):
  """
  Parameters
  ----------
  hypothesis : Scaling hypothesis.
  parameters : Parameters for the scaling hypothesis.
  data       : Data to evaluate the scaling hypothesis against.

  Returns
  -------
  S : The cost function.
  """
  def chi_square(parameters, L, data_L):
    # Get the scaled data points and errors
    x, y, dy = data_L[:,0], data_L[:,1], data_L[:,2]
    x, y, dy = hypothesis.scale(parameters, L, (x, y, dy))

    # Evaluate the master curve at the data points and get it's standard error
    Y, dY = master_curve(hypothesis, parameters, data, x)

    # Calculate the chi-square summand for these data points
    return ((y-Y) * (y-Y)) / (dy*dy + dY*dY)

  chis = [chi_square(parameters, data_L['L'], data_L['data'])
          for data_L in data]
  with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
      return np.nanmean(np.concatenate(chis ,axis=0))
    except RuntimeWarning:
      return np.finfo(np.float).max

def getBrackets(f, middle):
  """
  Find bracketing intervals for the nearest roots of the objective function.

  Parameters
  ----------
  f      : Objective function around the minimum found.
  middle : Argument which minimises the S+1 function.

  Returns
  -------
  lBracket : Bracket containing the left root.
  rBracket : Bracket containing the right root.
  """
  factor = 0.1 # Constant used to modify the bracketing interval
  
  f_mid = f(middle)
  left = middle * (1 - factor * np.sign(middle))
  f_left = f(left)
  while f_left * f_mid > 0.:
    left = (1. + factor) * left - factor * middle
    f_mid = f_left
    f_left = f(left)
    log.debug('LBracket: left=%s', left)
  lBracket = (left, middle)

  f_mid = f(middle)
  right = middle * (1 + factor * np.sign(middle))
  f_right = f(right)
  while f_right * f_mid > 0.:
    right = (1. + factor) * right - factor * middle
    f_mid = f_right
    f_right = f(right)
    log.debug('RBracket: right=%s', right)
  rBracket = (middle, right)

  return lBracket, rBracket

def errorAnalysis(hypothesis, parameters, data, min_S):
  """
  Parameters
  ----------
  hypothesis : Scaling hypothesis.
  parameters : Arguments which minimise the cost function.
  data       : Data for the cost function.
  min_S      : Minimum of the cost function.

  Returns
  -------
  errors : Dictionary of errors for the parameters provided.
  """
  def helper(key, value):
    def f(v):
      x = parameters.copy()
      x[key] = v
      return cost(hypothesis, x, data) - (min_S+1.)

    # Find bracketing intervals for the roots of the S+1 objective function.
    left, right = getBrackets(f, value)

    # Use a bisection search to find the left and right roots within the
    # intervals.
    left = opt.bisect(f, *left, xtol=1.0e-5)
    right = opt.bisect(f, *right, xtol=1.0e-5)

    return (key, (abs(left - value), abs(right - value)))

  return dict([helper(key, value) for key, value in parameters.iteritems()])
