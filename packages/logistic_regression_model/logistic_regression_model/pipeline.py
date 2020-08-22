from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

from feature_engine import categorical_encoders as ce
from feature_engine import variable_transformers as vt

from logistic_regression_model.processing import preprocessors as pp
from logistic_regression_model.config import config

import logging


_logger = logging.getLogger(__name__)


loan_pipe = Pipeline([
    
    # categorical missing value imputer
    ('categorical_imputer', 
        pp.CategoricalImputer(variables=config.VARS_WITH_NA)),
    
    # frequent label categorical encoder
    ('rare_encoder', 
        ce.RareLabelCategoricalEncoder(tol=0.02, 
                                        n_categories=4,
                                        variables=config.VARS_WITH_RARE,
                                        replace_with='Other', 
                                        return_object=True)),

    # target guided ordinal categorical variable encoder
    ('ordinal_encoder', 
        ce.OrdinalCategoricalEncoder(encoding_method='ordered',
                                     variables=config.ORDINAL_VARS)),
    
    # nominal categorical variable encoder (one hot)
    ('nominal_encoder', 
        ce.OneHotCategoricalEncoder(variables=config.NOMINAL_VARS,
                                    drop_last=True)),
        
    # Yeo-Johnson numerical variable transformer
    ('yeo_johnson_transformer', 
        vt.YeoJohnsonTransformer(variables=config.NUMERICAL_VARS)),
    
    # scaler
    ('min_max_scaler', MinMaxScaler()),
    
    # logistic regression classifier
    ('log_classifier', 
        LogisticRegression(class_weight='balanced', 
                          random_state=config.RANDOM_STATE, 
                          n_jobs=-1))
])