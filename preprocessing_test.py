from pathlib import Path
import pandas as pd

import data_pipeline


if __name__ == "__main__":
    
    data_dir = Path("./raw.csv")
    data = pd.read_csv(data_dir, parse_dates=["time"], index_col="time")
    
    
    range_limits = {
        'max_num': {
            'INT_CO2': 870, 'INT_REL_HD': 105, 'MID_INT_TMP': 40,'SLR_RAD': 900
        },
        'min_num': {
            'INT_CO2': 290, 'INT_REL_HD': 15, 'MID_INT_TMP': 9,'SLR_RAD': 0
        }
    }
    expected_types = {'INT_CO2': int, 'INT_REL_HD': float, 'MID_INT_TMP': float, 'SLR_RAD' : float}
    error_values = {"INT_CO2": [9999, -9999],  "MID_INT_TMP": [9999, -9999]}
    
    data_type = 'DF'
    pipeline_example = [
        ['data_refinement', 
            {'remove_duplication': {'flag': True}, 
            'static_frequency': {'flag': True, 'frequency': None}}],
        ['data_outlier',
            {'certain_error_to_NaN': {'data_min_max_limit': range_limits, 'flag': True},
            'uncertain_error_to_NaN': {'flag': True,
                'param': {'outlierDetectorConfig': 
                        [{'algorithm': 'IQR',
                                'percentile': 95,
                                'alg_parameter': {'weight':80}
                            }]}}}],
        ['data_imputation',
            {'flag': True,
            'imputation_method': [{'min': 0,'max': 20000,'method': 'linear','parameter': {}}],
            'total_non_NaN_ratio': 1}],
        ]
    
    valid_flag = data_pipeline.pipeline_connection_check(pipeline_example, data_type)
    
    if valid_flag:
        processing_data = data_pipeline.pipeline(data, pipeline_example, expected_types=expected_types)
    else:
        print("It's not working")
        
    processing_data.to_csv('processed_data.csv')
        
    