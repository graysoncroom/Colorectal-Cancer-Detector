# ==== System Imports ====
import os
import random
from datetime import datetime
import argparse

# ==== External Library Imports ====
from PIL import Image, ImageOps
from sklearn.pipeline import Pipeline, make_union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# ==== Giotto-tda Imports ====
from gtda.images import Binarizer, DensityFiltration, RadialFiltration, ImageToPointCloud, DensityFiltration, HeightFiltration
from gtda.homology import CubicalPersistence, VietorisRipsPersistence
from gtda.plotting import plot_heatmap, plot_diagram
from gtda.diagrams import Scaler, Amplitude, NumberOfPoints, PersistenceEntropy, BettiCurve


# ==== Lang Helper Imports ====
from functools import reduce
from typing import Callable
from typing import Any

# ==== Constants ====
DATA_DIR           : str         = 'Colorectal-Histology-MNIST/'
MODEL_SAVE_DIR     : str         = 'models/'
TILED_IMG_DATA_DIR : os.PathLike = os.path.join(DATA_DIR, 'Kather_texture_2016_image_tiles_5000/'*2)
FULL_IMG_DATA_DIR  : os.PathLike = os.path.join(DATA_DIR, 'Kather_texture_2016_larger_images_10/'*2)

TRAIN_SIZE_BINARY_SETTING, TEST_SIZE_BINARY_SETTING = 480, 120
TRAIN_SIZE_3LABEL_SETTING, TEST_SIZE_3LABEL_SETTING = 810, 90


# ==== Helper Functions ====

# DESC       : Recursively prints all files in a given directory
# EX         : print_contained_files(DATA_DIR); print_contained_files(TILED_IMG_DATA_DIR)
# MOD DATE   : Oct 23, 2022
def print_contained_files(directory_path: os.PathLike) -> None:
    list_of_dirs: List[os.PathLike] = []
    for f in os.listdir(directory_path):
        full_file_path = os.path.join(directory_path, f)
        if os.path.isfile(full_file_path):
            print(f'{f}\n')
        elif os.path.isdir(full_file_path):
            list_of_dirs += [full_file_path]
        else:
            print(f'ALERT: {f} is neither a file nor a directory', file=sys.stderr)
    if 0 != len(list_of_dirs):
        for d in list_of_dirs:
            print(f'====== The following are contained in {d} ======\n')
            # recursive call of this function on all directories in given directories
            print_contained_files(d) 

# DESC     : Compose an arbitrary large (but finite) number of functions that each take only 1 arg
# EX       : compose(f, g)(x) == f(g(x))
# MOD DATE : Nov 5, 2022
def compose(*fns) -> Callable[[Any], Any]:
    def inner(arg):
        for f in reversed(fns):
            arg = f(arg)
        return arg
    return inner

# DESC     : Returns function waiting to take the second arg of os.path.join, given its first arg
# EX       : partial_path_join(dir)(file) == os.path.join(dir, file)
# MOD DATE : Nov 23, 2022
def partial_path_join(path_arg1: os.PathLike) -> Callable[[os.PathLike], os.PathLike]:
    return lambda path_arg2: os.path.join(path_arg1, path_arg2)

# DESC     : Print PIL images as subplots of a single window
# MOD DATE : Nov 1, 2022
def display_images(
        images, 
        labels,
        columns=5,
        width=20,
        height=8,
        max_images=15,
        label_wrap_length=50,
        label_font_size=8,
        cmap='viridis'):

    # Guard against images array not being there
    if len(images) == 0:
        print('ALERT: No times to display!', file=sys.stderr)
        return
    
    # Guard against # of images exceeding max_images
    if len(images) > max_images:
        print(f'ALERT: Only showing {max_images} / {len(images)}')
        images = images[0:max_images]

    # height = max{height, floor( (# of images) / (# of columns) ) * height}
    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))

    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns) + 1, columns, i + 1)
        plt.imshow(image, cmap=cmap)

        plt.title(labels[i], fontsize=label_font_size)
    plt.show()

# DESC     : Converts np array of image to normal array of PIL images
# MOD DATE : Nov 1, 2022
def fromnp_to_image(np_images):
    image_images = []
    for img in np_images:
        image_images += [Image.fromarray(np.uint8(img))]
    return image_images

# DESC     : Abstracted Model Scoring and Result printing
# MOD DATE : Nov 13, 2022
def score_model_print_results(model_name: str, model, X_test, y_test, is_multiclass=False, plot_confusion=False):
    text = f'===== Results from {model_name} model ====' 
    print(text)
    #print(X_test[0])
    #print(model.score(np_heat_images_test))
    score = model.score(X_test, y_test)
    print(f'built-in model score: {score}')

    if is_multiclass:
        cm = confusion_matrix(y_test, model.predict(X_test))

        print('Confusion Matrix')
        print(f'TP: {cm[0][0]} FP: {cm[0][1]}')
        print(f'FN: {cm[1][0]} TN: {cm[1][1]}')

        print(classification_report(y_test, model.predict(X_test)))

        if plot_confusion:
            color = 'white'
            matrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
            matrix.ax_.set_title('Confusion Matrix', color=color)
            plt.xlabel('Predicted Label', color=color)
            plt.ylabel('True Label', color=color)
            plt.gcf().axes[0].tick_params(colors=color)
            plt.gcf().axes[1].tick_params(colors=color)
            plt.show()

    else:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f'Area under ROC Curve: {roc_auc}')

    print('='*len(text))

# DESC     : Used this to generated images for latex presentation
# MOD DATE : Nov 15, 2022
def generate_and_display_full_images_for_presentation():
    full_labels: List[os.PathLike] =  list(filter(compose(os.path.isfile,
                                                          partial_path_join(FULL_IMG_DATA_DIR)),
                                                  os.listdir(FULL_IMG_DATA_DIR)))
    full_imgs = []
    for lbl in full_labels:
        full_imgs += [Image.open(os.path.join(FULL_IMG_DATA_DIR, lbl))]

    display_images(full_imgs, full_labels)

# DESC      : add np_img and label to X_train and y_train respectively
# TYPE INFO : datasets is a list of tuples e.g. (X_train, y_train)
# MOD DATE  : Nov 18, 2022
def add_to_datasets(np_img, label, datasets):
    for X, y in datasets:
        X.append(np_img)
        y.append(label)

# DESC   : Calculates the best parameters for my models
# Ex ret : {'max_depth' : 10, 'n_estimators': 250}
def calculate_best_parameters(model, parameters, X_train, y_train):
    cv = GridSearchCV(model, parameters, cv=10, n_jobs=-1)
    cv.fit(X_train, y_train)
    return cv.best_params_

# DESC     : Collects all data, returns 4 parameters in order X, y, X, y
# MOD DATE : Oct 27, 2022
def collect_datasets(cell_classes):
    # binary setting data
    np_images_dataset1 = []
    y_dataset1 = []

    # 3-Label setting data
    np_images_dataset2 = []
    y_dataset2 = []

    list_dir = compose(os.listdir, partial_path_join(TILED_IMG_DATA_DIR))

    for label in cell_classes:
        # Simplify label (lowercase characters / no digits)
        # e.g. label = 03_LYMPHO -----> fixed_label = lympho
        fixed_label = (''.join([ch for ch in label if ch.isalpha()])).lower()

        if fixed_label == 'tumor' or fixed_label ==  'complex':
            datasets = [(np_images_dataset1, y_dataset1), (np_images_dataset2, y_dataset2)]
        elif fixed_label == 'lympho':
            datasets = [(np_images_dataset2, y_dataset2)]
        else:
            # This is not a data directory we care to include, so we skip it
            continue
            
        files_in_label_dir = list_dir(label)
        count = 0
        for f in files_in_label_dir:
            # Dr. Coskunuzer said dataset must be 300 images of each type (in respective setting)
            # in his email to me
            if count < 300:
                pil_img = Image.open(os.path.join(TILED_IMG_DATA_DIR, label, f))
                # Downsize pil_img with anti-alias filter (max quality)
                pil_img_small = pil_img.resize((75, 75), Image.ANTIALIAS)
                # convert PIL Image type to numpy array type
                np_image = np.asarray(ImageOps.grayscale(pil_img_small))[None, :, :]
                add_to_datasets(np_image, fixed_label, datasets)
                count += 1 # update counter

    return np_images_dataset1, y_dataset1, np_images_dataset2, y_dataset2

# e.g. params: { "n_estimators":[5, 10, 100, 250, 350], "max_depth":[2, 4, 8, 16, 32, None] }
def train_model(model_name_abv: str, setting: str, params, model_class, X_train, y_train):
    model = model_class()
    print(f'[{model_name_abv} {setting} Setting] Calculating best hyperparameters')
    best_params = calculate_best_parameters(model, params, X_train, y_train)
    print(f'[{model_name_abv} {setting} Setting] Best Hyperparameters: {best_params}')
    print(f'[{model_name_abv} {setting} Setting] Starting to train model with best hyperparameters')
    best_model = model_class(**best_params)
    best_model.fit(X_train, y_train)
    print(f'[{model_name_abv} {setting} Setting] Finished Training')
    return best_model

# DESC : Tool used to generate images for presentation
# Note : Image needs to be a NumPy array (matrix)
def generate_scaled_nonscaled_diagrams(image):
    cubical_persistence = CubicalPersistence(n_jobs=-1)                                       
    np_cubical_image = cubical_persistence.fit_transform(image)

    cubical_persistence.plot(np_cubical_image).show()

    scaler = Scaler()                                                                         
    np_scaled_image = scaler.fit_transform(np_cubical_image)
    scaler.plot(np_scaled_image).show()                                                         

# DESC: Where all the magic happens!
def main(model_to_import: str = None):
    # cell_classes will be a list of directory names w/o path
    # e.g. [03_LYMPHO, 08_EMPTY, ...]
    # NOTE: We need to convert to a list because the object returned by filter is 1 time use,
    cell_classes: List[os.PathLike] = list(filter(compose(os.path.isdir, 
                                                          partial_path_join(TILED_IMG_DATA_DIR)),
                                                  os.listdir(TILED_IMG_DATA_DIR)))

    # ===== Collecting Images into Variables =====
    np_images_dataset1, y_dataset1, np_images_dataset2, y_dataset2 = collect_datasets(cell_classes)

    # NOTE: We should now have np_images_dataset1 populated with 300 tumor and 300 complex stroma cell images
    #       and np_images_dataset2 populated with 300 tumor, complex stroma, and immune cell images

    # Some previously used steps. Here for future reference.
    #('binarizer', Binarizer()),
    #('filtration', RadialFiltration()),
    #('filtration', DensityFiltration(radius=3)),
    #('diagram', VietorisRipsPersistence()),
    #('betti', BettiCurve(n_bins=100, n_jobs=-1))

    steps = [
        ('diagram', CubicalPersistence()),
        ('rescaling', Scaler()),
        ('amplitude', Amplitude(metric='persistence_image'))
    ]
    pipeline = Pipeline(steps)

    # ==== Split up data and labels for training / testing ====
    X_train_binary_setting, X_test_binary_setting, y_train_binary_setting, y_test_binary_setting \
        = train_test_split(np_images_dataset1, y_dataset1, 
                           train_size=TRAIN_SIZE_BINARY_SETTING, test_size=TEST_SIZE_BINARY_SETTING, 
                           stratify=y_dataset1, random_state=666)

    X_train_3label_setting, X_test_3label_setting, y_train_3label_setting, y_test_3label_setting \
        = train_test_split(np_images_dataset2, y_dataset2, 
                           train_size=TRAIN_SIZE_3LABEL_SETTING, test_size=TEST_SIZE_3LABEL_SETTING, 
                           stratify=y_dataset2, random_state=666)

    fit_and_reshape = compose(lambda x: x.reshape(2), pipeline.fit_transform)

    # ==== Push Images through pipelines ====
    print('[Pipeline] About to start pushing images through heat pipeline')

    print('[Pipeline] Pushing Binary Setting Images')
    np_heat_images_train_binary_setting = list(map(fit_and_reshape, X_train_binary_setting))
    np_heat_images_test_binary_setting  = list(map(fit_and_reshape, X_test_binary_setting))

    print('[Pipeline] Pushing 3-Label Setting Images')
    np_heat_images_train_3label_setting = list(map(fit_and_reshape, X_train_3label_setting))
    np_heat_images_test_3label_setting = list(map(fit_and_reshape, X_test_3label_setting))


    # ==== Start Training the models ====
    if model_to_import == None:
        print('[INFO] About to start model training')
        print('[INFO] Results for each model will be printed at the end of all training')
        print('[INFO] This may take a while, but you will be notified after each model finishes training')

        rf_hyper_parameters = {
            "n_estimators": [5, 10, 100, 250, 350, 300, 450],
            "max_depth": [2, 3, 4, 8, 16, 32, None]
        }

        gb_hyper_parameters = {
            'n_estimators': [5, 50, 250, 500],
            'max_depth': [1, 3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 1, 10, 100]
        }

        rf_bin = train_model('RF', 'Binary', rf_hyper_parameters, RandomForestClassifier, np_heat_images_train_binary_setting, y_train_binary_setting)
        rf_3l = train_model('RF', '3-Label', rf_hyper_parameters, RandomForestClassifier, np_heat_images_train_3label_setting, y_train_3label_setting)
        gb_bin = train_model('GB', 'Binary', gb_hyper_parameters, GradientBoostingClassifier, np_heat_images_train_binary_setting, y_train_binary_setting)
        gb_3l = train_model('GB', '3-Label', gb_hyper_parameters, GradientBoostingClassifier, np_heat_images_train_3label_setting, y_train_3label_setting)
    else:
        print('[INFO] Importing given model')
        path = model_to_import
        rf_bin = joblib.load(os.path.join(path, 'rf_bin_model'))
        rf_3l = joblib.load(os.path.join(path, 'rf_3l_model'))
        gb_bin = joblib.load(os.path.join(path, 'gb_bin_model'))
        gb_3l = joblib.load(os.path.join(path, 'gb_3l_model'))

    # ==== Print Model Scores ====
    print('[INFO] Printing Model Results')
    score_model_print_results('Random Forest Binary Setting', rf_bin, np_heat_images_test_binary_setting, y_test_binary_setting)
    score_model_print_results('Random Forest 3-Label Setting', rf_3l, np_heat_images_test_3label_setting, y_test_3label_setting, is_multiclass=True)
    score_model_print_results('Gradient Boost Binary Setting', gb_bin, np_heat_images_test_binary_setting, y_test_binary_setting)
    score_model_print_results('Gradient Boost 3-Label Setting', gb_3l, np_heat_images_test_3label_setting, y_test_3label_setting, is_multiclass=True)

    # ==== Save Models ====
    # Comes after printing model scores so the user can decide if the models are good enough to be saved
    while model_to_import == None:
        user_input = str(input('Would you like to save these models? [y/n]: ')).lower()
        if user_input == 'y':
            print('[INFO] Saving Models to Disk')
            time: str = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
            path = os.path.join(MODEL_SAVE_DIR, time)
            os.mkdir(path)
            joblib.dump(rf_bin, os.path.join(path, 'rf_bin_model'))
            joblib.dump(rf_3l,  os.path.join(path, 'rf_3l_model'))
            joblib.dump(gb_bin, os.path.join(path, 'gb_bin_model'))
            joblib.dump(gb_3l, os.path.join(path, 'gb_3l_model'))
            break
        elif user_input == 'n':
            break
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorectal Histology MNIST Tissue Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--import-from-dir', action='store', type=str, help='import trained models from directory', const=None)
    args = parser.parse_args()
    config = vars(args)
    main(config['import_from_dir'])
