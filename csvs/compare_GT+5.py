import pandas as pd
import numpy as np

PROP_TRAIN_FILE = 'traversability_train_log.csv'
PREV_TRAIN_FILE = 'senko_traversability_train_log.csv'
OUTPUT_FILE = 'integrated_supervision_signals+5.csv'

def export_aligned_csvs():
    try:
        df_prop = pd.read_csv(PROP_TRAIN_FILE).sort_values('mission_timestamp')
        df_prev = pd.read_csv(PREV_TRAIN_FILE).sort_values('mission_timestamp')

        raw_metrics_cols = [
            'gt_slip', 
            'gt_imu_rp', 
            'gt_imu_gyro', 
            'gt_wheel_speed', 
            'gt_wheel_accel'
        ]

        aligned_all = pd.merge_asof(
            df_prop, 
            df_prev[['mission_timestamp', 'true_label']], 
            on='mission_timestamp', 
            direction='nearest', 
            tolerance=0.1,
            suffixes=('_prop', '_prev')
        ).dropna()

        columns_to_extract = [
            'mission_timestamp', 
            'true_label_prop', 
            'true_label_prev', 
            'traversability_cost'
        ] + raw_metrics_cols

        integrated_output = aligned_all[columns_to_extract].rename(columns={
            'mission_timestamp': 'ts', 
            'true_label_prop': 'ss_proposed', 
            'true_label_prev': 'ss_previous',
            'traversability_cost': 'dataset_gt'
        })

        integrated_output.to_csv(OUTPUT_FILE, index=False)

    except Exception:
        pass

if __name__ == "__main__":
    export_aligned_csvs()
