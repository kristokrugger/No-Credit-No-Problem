from logistic_regression_model.config import config as model_config
from logistic_regression_model.processing.data_management import load_dataset
from logistic_regression_model import __version__ as _version

import json

from api import __version__ as api_version


def test_home_endpoint_returns_200(flask_test_client):
    response = flask_test_client.get('/')

    assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
    response = flask_test_client.get('/version')

    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version
