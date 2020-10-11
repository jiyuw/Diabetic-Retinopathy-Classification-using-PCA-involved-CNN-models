from tensorflow.keras import applications as APP
from sources import util

def model_assessment(name, model_bases):
    # create model
    model_name = name + '_Base'
    model_base = model_bases[model_name](include_top = False, weights = 'imagenet', input_shape = (224,224,3))
    
    # train model
    model = util.addTops(model_base, filter_num = 4096, dense_num = 2, class_num = 5)
    model = util.train_model(model, model_name, 
                             train_dir = '../input/dr-classification-new-preprocessed-data/preprocessed_images/train', 
                             test_dir = '../input/dr-classification-new-preprocessed-data/preprocessed_images/test',
                             save_dir = './', partial_training = True)
    
    # PCA analysis
    post_pca = util.model_PCA(model, model_name, mode = 1,
                   test_dir = '../input/dr-classification-new-preprocessed-data/preprocessed_images/test',
                   save_dir = './')
    
    # rebuild model using PCA results
    model_pca = util.addTops(model_base, filter_num = post_pca, dense_num = 2, class_num = 5)
    model_name_pca = name + '_PCA'
    
    # train the post PCA model
    model_pca = util.train_model(model_pca, model_name_pca, 
                             train_dir = '../input/dr-classification-new-preprocessed-data/preprocessed_images/train', 
                             test_dir = '../input/dr-classification-new-preprocessed-data/preprocessed_images/test',
                             save_dir = './', partial_training = True)

if __name__ == '__main__':
    # Models need to be checked
    model_bases = {'VGG16_Base': APP.VGG16,
                   'DenseNet169_Base': APP.DenseNet169,
                   'InceptionV3_Base': APP.InceptionV3,
                   'MobileNetV2_Base': APP.MobileNetV2,
                   'NASNetMobile_Base': APP.NASNetMobile,
                   'ResNet152_Base': APP.ResNet152,
                   'Xception_Base': APP.Xception}

    # Example: if check vgg16 model
    model_assessment('VGG16', model_bases)
