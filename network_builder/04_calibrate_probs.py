import ibis
import sklearn.isotonic as iso
import pandas as pd

def calibrate_probabilities(probabilities: ibis.Table, expected_degrees: ibis.Table) -> ibis.Table:
    """
    Calibrate the raw link probabilities to match the expected degree distribution.

    Parameters:
    - probabilities: An Ibis Table containing raw link probabilities (p_ij) for each candidate edge.
    - expected_degrees: An Ibis Table containing the expected degree for each enterprise, by sector and size class.

    Returns:
    - calibrated_probabilities: An Ibis Table with calibrated probabilities that align with the expected degree distribution.

    Note: This function uses isotonic regression for calibration, which ensures that the calibrated probabilities are non-decreasing with respect to the raw probabilities.
    """
    # Convert Ibis Tables to Pandas DataFrames for processing
    prob_df = probabilities.to_pandas()
    degree_df = expected_degrees.to_pandas()

    # Merge the probabilities with the expected degrees based on sector and size class
    merged_df = prob_df.merge(degree_df, on=['sector', 'size_class'], how='left')
    # Perform isotonic regression for each sector and size class
    calibrated_probs = []
    for (sector, size_class), group in merged_df.groupby(['sector', 'size_class']):
        iso_reg = iso.IsotonicRegression(out_of_bounds='clip')
        calibrated = iso_reg.fit_transform(group['raw_probability'], group['expected_degree'])
        calibrated_probs.append(pd.DataFrame({
            'enterprise_id': group['enterprise_id'],
            'supplier_id': group['supplier_id'],
            'calibrated_probability': calibrated
        }))
    # Concatenate all calibrated probabilities into a single DataFrame
    calibrated_df = pd.concat(calibrated_probs, ignore_index=True)
    # Convert the DataFrame back to an Ibis Table
    calibrated_table = ibis.pandas.from_dataframe(calibrated_df)
    return calibrated_table