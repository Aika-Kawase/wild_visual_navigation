import rosbag
import rospy

# --- 設定 ---
ORIGINAL_BAG_PATH = '../../dataset_2/slag_heap_2023-09-14-12-23-05/slag_heap_1.bag'
OUTPUT_BAG_PATH = '../../dataset_2/slag_heap_2023-09-14-12-23-05/slag_heap_1_combined_loop_2.bag'
NUM_LOOPS = 2  # 結合する総周回数 (例: 2周分)
# -----------

def combine_and_offset_bag_fixed(original_path, output_path, num_loops):
    
    # 1. Bagの最初と最後の時刻を特定し、オフセットを計算
    latest_msg_time = rospy.Time(0)
    earliest_msg_time = rospy.Time.from_sec(9999999999.0) # 非常に大きな値で初期化
    
    with rosbag.Bag(original_path, 'r') as bag_r:
        for topic, msg, t in bag_r.read_messages():
            # ヘッダー時刻ではなく、Bagの記録時刻 (t) のMin/Maxを使う
            if t > latest_msg_time:
                latest_msg_time = t
            if t < earliest_msg_time:
                earliest_msg_time = t
    
    # 1周目のBagの実際の長さ (Duration)
    bag_duration = (latest_msg_time - earliest_msg_time).to_sec()
    
    # メッセージヘッダーに加算する絶対時刻オフセット。1周目の終了時刻の絶対値を使う。
    absolute_time_offset = latest_msg_time.to_sec() 
    
    print(f"1周目のBag duration: {bag_duration:.2f} 秒")
    print(f"絶対時刻オフセット (2周目以降の加算値): {absolute_time_offset:.2f} 秒")
    
    # 2. Bagの結合とオフセットの適用
    current_abs_offset = 0.0 # メッセージヘッダーに加算する絶対オフセット
    current_rec_offset = 0.0 # Bag記録時刻 (t) に加算する相対オフセット
    
    with rosbag.Bag(output_path, 'w') as bag_w:
        
        for loop_num in range(1, num_loops + 1):
            
            print(f"\n--- ループ {loop_num} の処理開始 ---")
            
            # オフセットを適用
            if loop_num > 1:
                # 2周目以降は、1周目の終了時刻を絶対時刻オフセットとして使用
                current_abs_offset += absolute_time_offset
                # 記録時刻 (t) には、durationの整数倍を加算
                current_rec_offset += bag_duration

            print(f"絶対時刻オフセット (Header): {current_abs_offset:.2f} 秒")
            print(f"記録時刻オフセット (Bag t): {current_rec_offset:.2f} 秒")
                
            with rosbag.Bag(original_path, 'r') as bag_r:
                # Bagを読み込み、時刻を調整して書き込み
                for topic, msg, t_orig in bag_r.read_messages():

                    # 1. 新しいBag記録時刻を計算 (Durationに基づいた相対オフセットを加算)
                    t_record_new = t_orig + rospy.Duration(current_rec_offset)
                    
                    # 2. メッセージヘッダーのタイムスタンプを計算 (絶対時刻オフセットを加算)
                    if topic == '/clock':
                        # 💡 修正点: /clock メッセージの 'clock' フィールドをオフセット
                        t_clock_new = msg.clock + rospy.Duration(current_abs_offset)
                        msg.clock = t_clock_new
                        # /clock メッセージは header を持たないので、t_header_newの計算は不要
                    elif hasattr(msg, 'header') and msg.header.stamp.to_sec() > 0:
                        # header を持つ一般的なメッセージを処理
                        t_header_new = msg.header.stamp + rospy.Duration(current_abs_offset)
                        msg.header.stamp = t_header_new
                    
                    # 調整後のメッセージを新しいBagに書き込む
                    bag_w.write(topic, msg, t_record_new)
                    
            print(f"ループ {loop_num} 完了。")
            
    print(f"\n✅ 結合済みBagファイル {output_path} が正常に作成されました。")

if __name__ == '__main__':
    # rospyを初期化 (ROSの時刻型を使用するため必須)
    rospy.init_node('bag_combiner_fixed', anonymous=True)
    combine_and_offset_bag_fixed(ORIGINAL_BAG_PATH, OUTPUT_BAG_PATH, NUM_LOOPS)