import argparse
import torch
import os

class BaseOptions():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    def initialize(self):
        self.parser.add_argument("--device", type=str, default="cuda:0")
        self.initialized = True
    def parse(self):
        self.initialize()
        return self.parser.parse_args()