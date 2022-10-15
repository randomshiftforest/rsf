import numpy as np

import sys
sys.path.append('autoperiod/src/autoperiod/')
from autoperiod import Autoperiod


def p(vs):
    values = np.array(vs)
    times = np.arange(1, values.size + 1)
    period = Autoperiod(times, values).period or 1
    return period
