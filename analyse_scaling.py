import numpy as np
import scipy as sp
import scipy.optimize as opt
import warnings
import logging
from collapseit import ScalingHypothesis, DefaultScalingHypothesis
from collapseit import master_curve, cost, getBrackets, errorAnalysis

# Setup logging system for this module
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

if __name__ == '__main__':
  hypothesis = DefaultScalingHypothesis()

  # Load the test data set
  data = [{'L':L, 'data':np.genfromtxt("data/orderParam_L{}.dat".format(L))}
          for L in [128, 256, 512]]

  # Evaluate the cost function for the data at a point.
  c_0 = {'x_c': 0.59, 'a': 0.75, 'b': 0.11}
  print cost(hypothesis, c_0, data)

  # Adaptor between the cost function to the scipy optimisation interface
  def cost_adaptor(params):
    return cost(hypothesis, hypothesis.unpack(params), data)

  # Find the (global) minimum of the cost function
  opt_res = opt.minimize(cost_adaptor, hypothesis.pack(c_0),
                         method='Nelder-Mead', options={'disp': True})
  """
  opt_res = opt.basinhopping(cost_adaptor, hypothesis.pack(c_0),
                             stepsize=0.1,
                             minimizer_kwargs={'method': 'Nelder-Mead'},
                             disp=True)
  """
  print opt_res

  c = c_0
  c = {'x_c': opt_res.x[0], 'a': opt_res.x[1], 'b': opt_res.x[2]}

  # Estimate the errors in the scaling analysis
  print "Error analysis"
  print errorAnalysis(hypothesis, c, data, opt_res.fun)

  import matplotlib as mpl
  mpl.use('TkAgg')  
  import matplotlib.pyplot as plt

  f, axs = plt.subplots(1,3)
  for data_L in data:
    # Plot the unscaled data_L
    axs[0].plot(data_L['data'][:,0], data_L['data'][:,1], label='$L={}$'.format(data_L['L']))

    # Plot the collapsed data
    x,y,dy = data_L['data'][:,0], data_L['data'][:,1], data_L['data'][:,2]
    x,y,dy = scale(c, data_L['L'], (x,y,dy))
    axs[1].set_title('opt')
    axs[1].plot(x, y, label='$L={}$'.format(data_L['L']))

    # Plot the collapsed data
    x,y,dy = data_L['data'][:,0], data_L['data'][:,1], data_L['data'][:,2]
    x,y,dy = scale(c_0, data_L['L'], (x,y,dy))
    axs[2].set_title('initial')
    axs[2].plot(x, y, label='$L={}$'.format(data_L['L']))
  plt.show()
