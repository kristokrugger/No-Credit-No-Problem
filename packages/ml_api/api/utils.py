from logistic_regression_model.predict import make_prediction

from api.config import get_logger

_logger = get_logger(logger_name=__name__)


def process_application(*, input_data) -> int:
    """The loan amount that the applicant is approved for
    is determined based on the probability score of the
    prediction.
    probability > 0.9: approved for $100K
    probability > 0.8: approved for $60K
    probability > 0.6: approved for $30K
    probability > 0.4: approved for $10K
    probability > 0.2: approved for $2K
    Otherwise approbed for $1000.
    """
    result = make_prediction(input_data=input_data, proba=True)
    _logger.debug(f'Outputs: {result}')
    probability = result['predictions'][0]

    base_amount = 1000

    if probability > 0.9: 
        return 100 * base_amount
    elif probability > 0.8: 
        return 60 * base_amount
    elif probability > 0.6: 
        return 30 * base_amount
    elif probability > 0.4: 
        return 10 * base_amount
    elif probability > 0.2: 
        return 2 * base_amount
    
    return base_amount
