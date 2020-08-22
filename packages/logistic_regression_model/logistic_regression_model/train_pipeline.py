import numpy as np
from sklearn.model_selection import train_test_split

from logistic_regression_model import pipeline
from logistic_regression_model.processing.data_management import (load_dataset, 
                                                                save_dataset, 
                                                                save_pipeline)
from logistic_regression_model.config import config
from logistic_regression_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # tranform the target with 0 and 1
    data[config.TARGET] = np.where(data[config.TARGET]=='>50K', 1, 0)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], 
        test_size=0.2, 
        random_state=config.RANDOM_STATE) 


    pipeline.loan_pipe.fit(X_train[config.FEATURES], y_train)

    save_dataset(data=X_test[config.FEATURES], 
        file_name=config.TESTING_DATA_FILE)

    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.loan_pipe)


if __name__ == "__main__":
    run_training()
