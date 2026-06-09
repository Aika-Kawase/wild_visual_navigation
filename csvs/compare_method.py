import pandas as pd
import numpy as np

PROP_GT_FILE = 'all_gt_log.csv'
PROP_TRAIN_FILE = 'traversability_train_log.csv'
PREV_GT_FILE = 'senko_all_gt_log.csv'
PREV_TRAIN_FILE = 'senko_traversability_train_log.csv'

def export_split_csvs():
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
        )
        prop_output = merged_prop[['mission_timestamp', 'gt_final_weighted', 'predicted_score', 'traversability_cost']].rename(
            columns={'mission_timestamp': 'ts', 'gt_final_weighted': 'self_supervision', 'predicted_score': 'predicted'}
        )
        prop_output.to_csv('proposed_method_data.csv', index=False)

        merged_prev = pd.merge_asof(
            df_prev_train, 
            df_prev_gt[['mission_time', 'gt_final_weighted']], 
            left_on='mission_timestamp', 
            right_on='mission_time',
            direction='nearest',
            tolerance=0.1
        )
        prev_output = merged_prev[['mission_timestamp', 'gt_final_weighted', 'predicted_score', 'traversability_cost']].rename(
            columns={'mission_timestamp': 'ts', 'gt_final_weighted': 'self_supervision', 'predicted_score': 'predicted'}
        )
        prev_output.to_csv('previous_method_data.csv', index=False)

    except FileNotFoundError:
        pass

if __name__ == "__main__":
    export_split_csvs()
