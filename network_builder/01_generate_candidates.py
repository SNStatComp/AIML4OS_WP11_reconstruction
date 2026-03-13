import ibis
import pandas as pd
import lightgbm as lgb
import ibis

def derive_dyadic_properties(users : ibis.Table, suppliers : ibis.Table) -> ibis.Table:
    # derive dyadic properties for all enterprise-supplier pairs, based on the attributes of the enterprises and suppliers

    pass

def generate_candidates(users : ibis.Table, suppliers : ibis.Table) -> ibis.Table:
    pass

def generate_candidates_probabilistic(users : ibis.Table, suppliers : ibis.Table) -> ibis.Table:
    # generate candidate links between enterprises and suppliers, based on the probabilities of supplier selection, by sector and size class, derived from the survey data and calibrated to match the expected number of suppliers per enterprise, by sector and size class
    pass

if __name__ == "__main__":
    # load data
    enterprises = ibis.read_parquet("data/enterprises.parquet")
    suppliers = ibis.read_parquet("data/suppliers.parquet")
    # derive dyadic properties
    dyadic_properties = derive_dyadic_properties(enterprises, suppliers)
    # generate candidate links
    candidate_links = generate_candidates(enterprises, suppliers)
    pass

