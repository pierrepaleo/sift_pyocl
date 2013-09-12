"""
Module Sift for calculating SIFT keypoint using PyOpenCL
"""
version = "0.2.0"
import os
sift_home = os.path.dirname(os.path.abspath(__file__))
import sys, logging
logging.basicConfig()
from .plan import SiftPlan
from .match import MatchPlan
from .alignment import LinearAlign

