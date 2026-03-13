import ibis

def estimate_expected_suppliers_per_user(users: ibis.Table, supplier_distribution: ibis.Table) -> ibis.Table:
    """
    Estimate the expected number of suppliers per enterprise, by sector and size class.

    Parameters:
    - users: An Ibis Table representing the users in the network.
    - supplier_distribution: An Ibis Table containing the distribution of suppliers per enterprise from the survey data.

    Returns:
    - expected_suppliers: An Ibis Table with the expected number of suppliers per enterprise, by sector and size class.
    """

    suppliers_count = users.groupby(['sector', 'size_class']).count().execute()
    survey_distribution = supplier_distribution.groupby(['sector', 'size_class']).agg({'num_suppliers': 'mean'}).execute()
    expected_suppliers = suppliers_count.merge(survey_distribution, on=['sector', 'size_class'])
    expected_suppliers['expected_num_suppliers'] = expected_suppliers['count'] * expected_suppliers['num_suppliers'] / expected_suppliers['count'].sum()
    return expected_suppliers[['sector', 'size_class', 'expected_num_suppliers']]
