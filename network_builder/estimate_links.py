import joblib
import lightgbm as lgb

def load_lightgbm_model() -> lgb.LGBMClassifier:
    model = joblib.load("models/model_LightGBM.pkl", mmap_mode="r")
    return model