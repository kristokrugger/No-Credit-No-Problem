import numpy as np

from logistic_regression_model.predict import make_prediction
from logistic_regression_model.processing.data_management import load_dataset


def test_make_single_prediction():
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')

    subject = make_prediction(input_data=single_test_json, proba=False)

    assert subject is not None
    assert len(subject.get('predictions')) == 1
    assert isinstance(subject.get('predictions')[0], np.int64)
    assert subject.get('predictions')[0] == 0

def test_make_single_prediction_proba():
    test_data = load_dataset(file_name='test.csv')
    single_test_json = test_data[0:1].to_json(orient='records')

    subject = make_prediction(input_data=single_test_json, proba=True)

    assert subject is not None
    assert len(subject.get('predictions')) == 1
    assert isinstance(subject.get('predictions'), np.ndarray)
    assert subject.get('predictions')[0] == 0.426


def test_make_multiple_predictions():
    test_data = load_dataset(file_name='test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    subject = make_prediction(input_data=multiple_test_json, proba=False)

    assert subject is not None
    assert len(subject.get('predictions')) == 9769
    assert np.max(subject.get('predictions')) == 1
    assert np.min(subject.get('predictions')) == 0

def test_make_multiple_predictions_proba():
    test_data = load_dataset(file_name='test.csv')
    original_data_length = len(test_data)
    multiple_test_json = test_data.to_json(orient='records')

    subject = make_prediction(input_data=multiple_test_json, proba=True)

    assert subject is not None
    assert len(subject.get('predictions')) == 9769
    assert np.max(subject.get('predictions')) == 0.997
    assert np.min(subject.get('predictions')) == 0.001

