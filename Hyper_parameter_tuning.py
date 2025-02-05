import os
import random
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlguess.keras.models import EvidentialRegressorDNN
from mlguess.regression_metrics import regression_metrics
from mlguess.keras.callbacks import get_callbacks



try:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
except ImportError as err:
    print("This example script requires tensorflow to be installed. Please install tensorflow before proceeding.")
    raise err

from echo.src.base_objective import BaseObjective
from optuna.integration import TFKerasPruningCallback
import optuna

import warnings

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)


def seed_everything(seed=1234):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()




class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss"):

        # Initialize the base class
        BaseObjective.__init__(self, config, metric)

    def train(self, trial,conf):

        learning_rate = conf["model"]["lr"]
        dropout_alpha =  conf["model"]["dropout_alpha"]
        hidden_layers = conf["model"]["hidden_layers"]
        batch_size = conf["model"]["batch_size"]
        l1_weight = conf["model"]["l1_weight"]
        l2_weight = conf["model"]["l2_weight"]
        hidden_neurons = conf["model"]["hidden_neurons"]
        activation = conf["model"]["activation"]
        evidential_coef = conf["model"]["evidential_coef"]
        epochs = conf["model"]["epochs"]
        seed = conf["seed"]

        # Fix seed for reproducibility
        seed_everything(seed)

        # Load the dataseT        
        train_data=pd.read_csv(
            "/glade/u/home/ijahan/notebook_evidential_IJ/Gridded_preds/Data/train_data.csv",
            sep=','
        )

        valid_data=pd.read_csv(
            "/glade/u/home/ijahan/notebook_evidential_IJ/Gridded_preds/Data/valid_data.csv",
            sep=','
        )
        
        met_vars = conf["met_vars"]
        output_vars = conf["output_vars"]
        
        # for standardization
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()


        x_train = scaler_x.fit_transform(train_data[met_vars])
        x_valid = scaler_x.transform(valid_data[met_vars])

        y_train = scaler_y.fit_transform(train_data[output_vars])
        y_valid = scaler_y.transform(valid_data[output_vars])


        #Load the model
        model = EvidentialRegressorDNN(**conf["model"])
        model.build_neural_network(x_train.shape[-1], y_train.shape[-1])
        model.fit(x_train,y_train,
                          validation_data=(x_valid,y_valid),
                          callbacks=get_callbacks(conf,path_extend="models"))
        history = model.model.history

        
        mu, ale, epi = model.predict_uncertainty(x_valid, scaler = scaler_y)
        total = np.sqrt(ale + epi)
        val_metrics = regression_metrics(scaler_y.inverse_transform(y_valid), mu, total)
        
        # I am not using val_loss as the optization metric, just wants to see the min val_loss of a trial
        min_val_loss = min(history.history['val_loss'])
        val_loss_dict = {'val_loss': min_val_loss}

        
        val_metrics.update(val_loss_dict)
        return val_metrics

        
