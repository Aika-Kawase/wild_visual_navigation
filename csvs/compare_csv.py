import pandas as pd
import numpy as np

PROP_GT_FILE = 'all_gt_log.csv'
PROP_TRAIN_FILE = 'traversability_train_log.csv'
PREV_GT_FILE = 'senko_all_gt_log.csv'
PREV_TRAIN_FILE = 'senko_traversability_train_log.csv'

def export_comparison_csv():
    try:
        df_prop_gt = pd.read_csv(PROP_GT_FILE).sort_values('mission_time')
        df_prop_train = pd.read_csv(PROP_TRAIN_FILE).sort_values('mission_timestamp')
        df_prev_gt = pd.read_csv(PREV_GT_FILE).sort_values('mission_time')
        df_prev_train = pd.read_csv(PREV_TRAIN_FILE).sort_values('mission_timestamp')

        merged_prop = pd.merge_asof(
            df_prop_train, 
            df_prop_gt[['mission_time', 'gt_final_weighted']], 
            left_on='mission_timestamp', 
            right_on='mission_time',
            direction='nearest',
            tolerance=0.1
        ).rename(columns={'gt_final_weighted': 'ss_prop', 'predicted_score': 'pred_prop', 'mission_timestamp': 'ts'})

        merged_prev = pd.merge_asof(
            df_prev_train, 
            df_prev_gt[['mission_time', 'gt_final_weighted']], 
            left_on='mission_timestamp', 
            right_on='mission_time',
            direction='nearest',
            tolerance=0.1
        ).rename(columns={'gt_final_weighted': 'ss_prev', 'predicted_score': 'pred_prev', 'mission_timestamp': 'ts'})

        final_df = pd.merge_asof(
            merged_prop[['ts', 'traversability_cost', 'ss_prop', 'pred_prop']],
            merged_prev[['ts', 'ss_prev', 'pred_prev']],
            on='ts',
            direction='nearest',
            tolerance=0.1
        ).dropna()

        final_df.to_csv('final_comparison_data.csv', index=False)

    except FileNotFoundError:
        pass

if __name__ == "__main__":
    export_comparison_csv()
