"""
Using ShallowNet.py to create the default shallow CNN models and training with the dataset
"""

from sources.DRC import DR_classifier
from sources.ShallowNet import ShallowNet


def main():
    # create a shallow CNN model
    # Use default filter map and top layers
    pre_model = ShallowNet()

    # Add this model to a DR classifier
    shallowDR = DR_classifier('ShallowCNN')