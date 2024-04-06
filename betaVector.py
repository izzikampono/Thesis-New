import numpy as np
from problem import PROBLEM
from alphaVector import AlphaVector


class BetaVector:
    """class representing beta vectors, state-action value vectors rooted at a fixed belief state of a game"""
    def __init__(self,two_d_vector_0,two_d_vector_1):
        self.two_d_vectors = [two_d_vector_0,two_d_vector_1]

    def print(self):
        print(self.two_d_vectors)