import numpy as np
import scipy as sp
import scipy.optimize as opt
import warnings
import logging
from abc import ABCMeta, abstractmethod

# Configure logging for this module
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class ScalingHypothesis(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def scale(self, parameters, L, (x,y,dy)):
    """
    Parameters
    ----------
    parameters : Parameters for the scaling hypothesis.
    L          : System size
    x          : Unscaled independent variable.
    y          : Unscaled dependent variable.
    dy         : Statistical error in unscaled dependent variable.

    Returns
    -------
    x  : Scaled independent variable.
    y  : Scaled dependent variable.
    dy : Statistical error in the scaled dependent variable.
    """
    raise NotImplementedError

  @abstractmethod
  def pack(self, parameters):
    """
    Pack the parameters for this scaling hypothesis.

    Parameters
    ----------
    parameters : Parameters for the scaling hypothesis as a dictionary.

    Returns
    -------
    parameters : The same parameters packed as a NumPy array.
    """
    raise NotImplementedError

  @abstractmethod
  def unpack(self, parameters):
    """
    Unpack the parameters for this scaling hypothesis.

    Parameters
    ----------
    parameters : Parameters for the scaling hypothesis packed as a NumPy array.

    Returns
    -------
    parameters : The same parameters unpacked as a dictionary.
    """
    raise NotImplementedError

class DefaultScalingHypothesis(ScalingHypothesis):
  """
  Implements the scaling hypothesis
  y = L^-b f( (x - x_c) * L^a)
  """
  def scale(self, parameters, L, (x,y,dy)):
    with warnings.catch_warnings():
      warnings.filterwarnings('error')
      try:
        x = (x - parameters['x_c']) * (L ** parameters['a'])
        y = y * (L ** parameters['b'])
        dy = dy * (L ** parameters['b'])
        return (x, y, dy)
      except RuntimeWarning, e:
        log.error("%s\nParameters: %s\n", e, parameters)
        raise

  def pack(self, parameters):
    return np.array([parameters['x_c'], parameters['a'], parameters['b']])

  def unpack(self, parameters):
    return {'x_c': parameters[0], 'a': parameters[1], 'b': parameters[2]}
