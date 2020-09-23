"""
Using ShallowNet.py to create variants of the base shallow CNN model
PCA info of base shallow:
    base filter map: [32, 32, 64, 64, 128, 128, 256, 256], dense layer [100, 100]
    var1 filter map: [23, 24, 37, 0, 125, 0, 254, 0], dense layer [100, 100]
    var2 filter map: [32, 32, 64, 64, 128, 128, 256, 256], dense layer [38, 44]
    var3 filter map: [23, 24, 37, 0, 125, 0, 254, 0], dense layer [38, 44]
"""


from sources import util
from sources.ShallowNet import ShallowNet


def main():
    # create shallow CNN variants
    var1 = ShallowNet(include_top=True, filter_map=[23, 24, 37, 0, 125, 0, 254, 0])
    var2 = ShallowNet(include_top=False)
    var3 = ShallowNet(include_top=False, filter_map=[23, 24, 37, 0, 125, 0, 254, 0])

    # add top for var2 and var3
    var2 = util.addTops(var2, filter_num=[38, 44])
    var3 = util.addTops(var3, filter_num=[38, 44])

    # train these models
    util.train_model(var1, 'ShallowNet_Var1')
    util.train_model(var2, 'ShallowNet_Var2')
    util.train_model(var3, 'ShallowNet_Var3')


if __name__ == '__main__':
    main()