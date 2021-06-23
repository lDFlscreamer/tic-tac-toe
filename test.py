import numpy as np
from numpy import unravel_index

import CONSTANT

q_s = np.random.normal(size=(CONSTANT.FIELD_SIZE, CONSTANT.FIELD_SIZE, 1))[:, :, 0]
a = unravel_index(q_s.argmax(), q_s.shape)
print("Sdsd")