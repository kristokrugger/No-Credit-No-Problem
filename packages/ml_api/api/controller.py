from flask import Blueprint, request, jsonify, render_template, url_for
from logistic_regression_model.predict import make_prediction
from logistic_regression_model.config import config
from logistic_regression_model import __version__ as _version

from api.config import get_logger
from api.utils import process_application
from api import __version__ as api_version

import json

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__,
                            template_folder='templates',
                            static_folder='static',
                            static_url_path='/%s' % __name__)


@prediction_app.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        _logger.debug('home status OK')
        return render_template('index.html')


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': _version,
                        'api_version': api_version})

@prediction_app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = dict(request.form)
        full_name = input_data['full_name']
        del input_data['full_name']
        del input_data['apply']
        for feature in config.NUMERICAL_VARS:
            input_data[feature] = int(input_data[feature])
        input_data = json.dumps([input_data])
        _logger.debug(input_data)

        approved_amount = process_application(input_data=input_data)

        outcome_msg = f"""
                        Congratulations, {full_name}! 
                        You have been approved for ${approved_amount}.
                        """

        
        return render_template('index.html', 
                                prediction_text=outcome_msg)

