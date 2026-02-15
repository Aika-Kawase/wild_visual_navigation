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
            df_prev[['mission_timestamp', 'true_label', 'predicted_score']], 
            on='mission_timestamp', 
            direction='nearest', 
            tolerance=0.1,
            suffixes=('_prop', '_prev')
        ).dropna()

        prop_output = aligned_all[[
            'mission_timestamp', 'true_label_prop', 'predicted_score_prop', 'traversability_cost'
        ]].rename(columns={
            'mission_timestamp': 'ts', 
            'true_label_prop': 'self_supervision', 
            'predicted_score_prop': 'predicted'
        })
        prop_output.to_csv('proposed_method_aligned.csv', index=False)

        prev_output = aligned_all[[
            'mission_timestamp', 'true_label_prev', 'predicted_score_prev', 'traversability_cost'
        ]].rename(columns={
            'mission_timestamp': 'ts', 
            'true_label_prev': 'self_supervision', 
            'predicted_score_prev': 'predicted'
        })
        prev_output.to_csv('previous_method_aligned.csv', index=False)

    except Exception:
        pass

if __name__ == "__main__":
    export_aligned_csvs()
