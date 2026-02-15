import pandas as pd
import numpy as np

PROP_TRAIN_FILE = 'traversability_train_log.csv'
PREV_TRAIN_FILE = 'senko_traversability_train_log.csv'

def export_aligned_csvs():
    try:
        df_prop = pd.read_csv(PROP_TRAIN_FILE).sort_values('mission_timestamp')
        df_prev = pd.read_csv(PREV_TRAIN_FILE).sort_values('mission_timestamp')

        aligned_all = pd.merge_asof(
            df_prop, 
            df_prev[['mission_timestamp', 'true_label']], 
            on='mission_timestamp', 
            direction='nearest', 
            tolerance=0.1,
            suffixes=('_prop', '_prev')
        ).dropna()

        integrated_output = aligned_all[[
            'mission_timestamp', 
            'true_label_prop', 
            'true_label_prev', 
            'traversability_cost'
        ]].rename(columns={
            'mission_timestamp': 'ts', 
            'true_label_prop': 'ss_proposed', 
            'true_label_prev': 'ss_previous',
            'traversability_cost': 'dataset_gt'
        })

        integrated_output.to_csv('integrated_supervision_signals.csv', index=False)

    except Exception:
        pass

if __name__ == "__main__":
    export_aligned_csvs()
