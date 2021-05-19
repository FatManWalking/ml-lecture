# +
from pathlib import path
import sys

sys.path.insert(0, os.path.abspath('..'))

from art.utils import load_nursery

(x_train, y_train), (x_test, y_test), _, _ = load_nursery(test_set=0.5)
