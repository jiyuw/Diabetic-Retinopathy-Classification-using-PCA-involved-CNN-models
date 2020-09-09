"""
Using ShallowNet.py to create the default shallow CNN models and training with the dataset
"""

from sources import util
from sources.ShallowNet import ShallowNet


def main():
    # create a shallow CNN model
    # Use default filter map and top layers
    pre_model = ShallowNet()
    model_name = 'ShallowNet_Base'

    # train this model
    pre_model = util.train_model(pre_model, model_name)

    # PCA analysis
    util.model_PCA(pre_model, model_name, mode=2)


if __name__ == '__main__':
    main()
