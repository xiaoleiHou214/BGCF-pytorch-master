import numpy as np
import pandas as pd
from wsdream.loaddata import PerformanceComput
userId = [i for i in range(30)]
userId += [i for i in range(30)]
userId += [i for i in range(30)]
pre = np.random.random_sample((90,))
lab = np.random.random_sample((90,))
perdiction = [i for i in pre]
label = [i for i in lab]
PerformanceComput(userId,label,perdiction)