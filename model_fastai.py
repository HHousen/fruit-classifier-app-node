from pathlib import Path
from fastai import *
from fastai.vision import *

# SETUP HERE
PKL_OR_PTH = 'pkl'
NAME_OF_FILE = 'export.pkl' # Name of your exported file
PATH_TO_MODELS_DIR = Path('models') # by default just use /models in root dir
if PKL_OR_PTH is 'pth':
    YOUR_CLASSES_HERE = ['black', 'grizzly', 'teddys'] # Your class labels

class FastaiImageClassifier(object):
    def __init__(self):
        if PKL_OR_PTH is 'pth':
            self.learner = self.setup_model_pth(PATH_TO_MODELS_DIR, NAME_OF_FILE, YOUR_CLASSES_HERE, normalizer=imagenet_stats)
        else:
            self.learner = self.setup_model_plk(PATH_TO_MODELS_DIR, NAME_OF_FILE)

    def setup_model_pth(self, path_to_pth_file, learner_name_to_load, classes, resnet_num=34, ds_tfms=get_transforms(), normalizer=None, **kwargs):
        "Initialize learner for inference"
        defaults.device = torch.device('cpu')
        data = ImageDataBunch.single_from_classes(path_to_pth_file, classes, ds_tfms=ds_tfms, **kwargs)
        if (normalizer is not None): data.normalize(normalizer)
        resnet = self.get_resnet(resnet_num)
        learn = create_cnn(data, resnet, pretrained=False)
        learn.load(learner_name_to_load)
        return learn

    def get_resnet(self, resnet_num=34):
        "Specify resnet: 18, 34, 50, 101, 152"
        return getattr(models, f'resnet{resnet_num}')

    def setup_model_plk(self, path_to_pkl_file, learner_name_to_load, **kwargs):
        defaults.device = torch.device('cpu')
        learn = load_learner(path_to_pkl_file, fname=learner_name_to_load)
        return learn

    def predict(self, img_path):
        img = open_image(Path(img_path))
        pred_class, pred_idx, losses = self.learner.predict(img)
        print('Class pred:', pred_class)
        print('Pred-idx:', pred_idx)
        print('Losses:', losses)
        return { 'predict': self.learner.data.classes[pred_idx] }
