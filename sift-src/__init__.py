version = "0.0.1"
import os
sift_home = os.path.dirname(os.path.abspath(__file__))
import sys, logging
logging.basicConfig()
from .plan import SiftPlan
