import joblib
import lightgbm as lgb
import ibis

def load_lightgbm_model() -> lgb.LGBMClassifier:
    model = joblib.load("models/model_LightGBM.pkl", mmap_mode="r")
    return model


def predict_raw_probabilities(candidates: ibis.Table, model: lgb.LGBMClassifier) -> ibis.Table:
    # derive dyadic features for all enterprise-supplier pairs
    # predict raw probabilities of link existence for all enterprise-supplier pairs
    raw_probabilities = model.predict_proba(candidates)[:, 1]  # Assuming binary classification and we want the probability of the positive class
    # combine with enterprise and supplier identifiers
    results = candidates[['user_id', 'supplier_id']].copy()
    results['raw_probability'] = raw_probabilities
    return results