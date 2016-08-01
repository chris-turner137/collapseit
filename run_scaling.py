import numpy as np
import analyse_scaling as analysis
import scipy.optimize as opt
from jsci.WriteStream import FileWriteStream as JSONWriter
from jsci.Coding import NumericDecoder
import sys
import jsci
import version
import json
import time
import warnings
import collapseit

t0 = time.time()

# Configure logging for this module
import logging
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stderr))
log_as = logging.getLogger('analyse_scaling')
log_as.setLevel(logging.DEBUG)
log_as.addHandler(logging.StreamHandler(sys.stderr))

class EntropyFluctuationsScalingHypothesis(analysis.ScalingHypothesis):
  """
  Implements the scaling hypothesis
  y = L^-b f( (x - x_c) * L^a)
  """
  def scale(self, parameters, L, (x,y,dy)):
    with warnings.catch_warnings():
      warnings.filterwarnings('once')
      return ((x - parameters['x_c']) * (L ** parameters['a']),
               y / (parameters['b'] +  L),
               dy / (parameters['b'] + L))

  def pack(self, parameters):
    return np.array([parameters['x_c'], parameters['a'], parameters['b']])

  def unpack(self, parameters):
    return {'x_c': parameters[0], 'a': parameters[1], 'b': parameters[2]}

class FixableScalingHypothesis(analysis.ScalingHypothesis):
  """
  Implements the scaling hypothesis
  y = L^-b f( (x - x_c) * L^a)
  """
  def __init__(self):
    self._fixed = {}
    return

  def set_fixed(self, parameters):
    self._fixed = parameters

  def get_fixed(self):
    return self._fixed

  def scale(self, parameters, L, (x,y,dy)):
    parameters = dict(parameters, **self._fixed)
    with warnings.catch_warnings():
      warnings.filterwarnings('once')
      return ((x - parameters['x_c']) * (L ** parameters['a']),
               y * (L ** parameters['b']),
               dy * (L ** parameters['b']))

  def pack(self, parameters):
    parameters = parameters.copy()
    for key in self._fixed.keys():
      parameters.pop(key, None)
    return np.array([parameters.values()])

  def unpack(self, parameters):
    # Build dictionary with keys which aren't fixed
    unpacked = {'x_c': 0, 'a': 0, 'b': 0}
    for key in self._fixed.keys():
      unpacked.pop(key, None)

    assert len(unpacked) == len(parameters)
    return dict(zip(unpacked.keys(), parameters))

# Configure the command line argument parser
import argparse
desc = """
Performs a finite size scaling analysis on data provided.
"""
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--c_0', dest='c_0', type=str,
                    help='Initial guess for scaling parameters.')
parser.add_argument('--c_min', dest='c_min', type=str,
                    help='Minimum values for scaling parameters.')
parser.add_argument('--c_max', dest='c_max', type=str,
                    help='Maximum values for scaling parameters.')
parser.add_argument('--c_fix', dest='c_fix', type=str,
                    help='Fixed values for scaling parameters.')
parser.add_argument('--method', dest='method', type=str, default='Hopping-Powell',
                    help='Method of finding the minimum.')
parser.add_argument('--hypothesis', dest='hypothesis', type=str, default='default',
                    help='Scaling hypothesis.')
parser.add_argument('--plot', action='store_true',
                    help='Plot the data and the collapse.')
args = parser.parse_args()

c_0 = eval(args.c_0) if args.c_0 is not None else {'x_c':0.59, 'a':0.75, 'b':0.11}
c_fix = eval(args.c_fix) if args.c_fix is not None else {}
c_min = eval(args.c_min) if args.c_min is not None else {'x_c':0.0}
c_max = eval(args.c_max) if args.c_max is not None else {'x_c':10.0}
if args.hypothesis == 'default':
  hypothesis = FixableScalingHypothesis()
elif args.hypothesis == 'entropy_fluctuations':
  hypothesis = EntropyFluctuationsScalingHypothesis()
else:
  raise ValueError, "Unrecognised scaling hypothesis."

# Remove fixed keys from the initial
for key in c_fix.keys():
  c_0.pop(key, None)
hypothesis.set_fixed(c_fix)

# Adaptor between the cost function and the scipy optimisation routines
def cost_adaptor(params):
  with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
      cost = analysis.cost(hypothesis, hypothesis.unpack(params), data)
    except Warning:
      cost = np.inf
  #log.debug('c: %s, f(c): %s', params, cost)
  return cost

# Load the data set from stdin
data = json.load(sys.stdin, cls=NumericDecoder)

out = JSONWriter(sys.stdout, indent=2)
with out.wrap_object():

  out.write_pair('git',
  {
    'imps': imps.version.get_version_string(),
    'collapseit': collapseit.get_version_string()
  })
  out.write_pair('method', args.method)
  out.flush()

  # Find the (global) minimum of the cost function
  if args.method == 'Nelder-Mead':
    opt_res = opt.minimize(cost_adaptor, hypothesis.pack(c_0),
                           method='Nelder-Mead', options={'maxfev':1000})
  else:
    def accept_test(f_new, x_new, f_old, x_old):
      params = hypothesis.unpack(x_new)
      log.debug('local minimum: c = %s, f(c) = %s', params, f_new)
      for k,v in params.iteritems():
        if (k in c_min and v < c_min[k]) or (k in c_max and v > c_max[k]):
          return False
      return True
    if args.method == 'Hopping-NM':
      opt_res = opt.basinhopping(cost_adaptor, hypothesis.pack(c_0),
                                 stepsize=1.0, T=5.0,
                                 accept_test=accept_test,
                                 minimizer_kwargs={'method': 'Nelder-Mead'},
                                 disp=False)
    elif args.method == 'Hopping-Powell':
      opt_res = opt.basinhopping(cost_adaptor, hypothesis.pack(c_0),
                                 stepsize=1.0, T=5.0,
                                 accept_test=accept_test,
                                 minimizer_kwargs={'method': 'Powell'},
                                 disp=False)
    elif args.method == 'Nelder-Mead':
      opt_res = opt.minimize(cost_adaptor, hypothesis.pack(c_0),
                             method='Nelder-Mead')
    else:
      raise ValueError, 'Unrecognised optimisation method.'

    if opt_res.message == ['requested number of basinhopping iterations'
                           ' completed successfully']:
      opt_res.success = True
    else:
      opt_res.success = False

  c = hypothesis.unpack(opt_res.x)

  log.debug('%s\n', opt_res.message)
  if not opt_res.success:
    exit(1)

  out.write_pair('nit', opt_res.nit)
  out.write_pair('nfev', opt_res.nfev)
  if hasattr(opt_res, 'status'):
    out.write_pair('status', opt_res.status)

  out.write_pair('min', opt_res.fun)
  out.write_pair('argmin', c)
  out.flush()

  # Estimate the errors in the scaling analysis
  errors = analysis.errorAnalysis(hypothesis, c, data, opt_res.fun)

  out.write_pair('errors', errors)
out.flush()

log.debug('Total time elapsed %s', time.time() - t0)

if args.plot:
  import matplotlib as mpl
  mpl.use('TkAgg')
  import matplotlib.pyplot as plt

  f, axs = plt.subplots(1,3)
  for data_L in data:
    # Plot the unscaled data_L
    x,y,dy = data_L['data'][:,0], data_L['data'][:,1], data_L['data'][:,2]
    axs[0].set_xscale('log')
    axs[0].errorbar(x, y, dy, label='$L={}$'.format(data_L['L']))

    # Plot the collapsed data
    x,y,dy = data_L['data'][:,0], data_L['data'][:,1], data_L['data'][:,2]
    x,y,dy = hypothesis.scale(c, data_L['L'], (x,y,dy))
    axs[1].set_title('opt')
    axs[1].errorbar(x, y, dy, label='$L={}$'.format(data_L['L']))

    # Plot the collapsed data
    x,y,dy = data_L['data'][:,0], data_L['data'][:,1], data_L['data'][:,2]
    x,y,dy = hypothesis.scale(c_0, data_L['L'], (x,y,dy))
    axs[2].set_title('initial')
    axs[2].errorbar(x, y, dy, label='$L={}$'.format(data_L['L']))
  plt.show()
