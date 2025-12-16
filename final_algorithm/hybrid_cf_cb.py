# ============================================================
# HYBRID RECOMMENDATION – INFERENCE USING TRAINED MODELS
# Combine Content-Based Filtering (CBF) + Collaborative Filtering (CF)
# For Custom Activity Dataset
# ============================================================

import sys
import os
import pickle
import importlib.util
import numpy as np
import pandas as pd

# Import các class cần thiết để pickle có thể load model
# Cần import trước khi load pickle
# Vì tên file có dấu gạch ngang, dùng importlib
current_dir = os.path.dirname(os.path.abspath(__file__))
cf_new_path = os.path.join(current_dir, 'cf-new.py')

spec = importlib.util.spec_from_file_location("cf_new", cf_new_path)
cf_new_module = importlib.util.module_from_spec(spec)
sys.modules['cf_new'] = cf_new_module
spec.loader.exec_module(cf_new_module)

# Import các class vào namespace hiện tại và __main__
# Điều này giúp pickle tìm thấy class khi load model
HybridCF = cf_new_module.HybridCF
CF = cf_new_module.CF

# Thêm vào __main__ namespace để pickle có thể tìm thấy
import __main__
__main__.HybridCF = HybridCF
__main__.CF = CF

# ============================================================
# 1. LOAD TRAINED MODELS
# ============================================================

CBF_MODEL_PATH = '../models/content_based_model.pkl'
CF_MODEL_PATH = '../models/hybrid_cf_model.pkl'

# Custom unpickler để xử lý mapping module names
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Nếu module là '__main__' hoặc 'cf_new', map về module đã import
        if module == '__main__' or module == 'cf_new' or 'cf-new' in module:
            if name == 'HybridCF':
                return HybridCF
            elif name == 'CF':
                return CF
        return super().find_class(module, name)

print('Đang tải models...')
with open(CBF_MODEL_PATH, 'rb') as f:
    cbf_data = pickle.load(f)
print('✅ Đã tải Content-Based model')

# Load CF model với custom unpickler để xử lý module mapping
try:
    with open(CF_MODEL_PATH, 'rb') as f:
        unpickler = CustomUnpickler(f)
        cf_data = unpickler.load()
    print('✅ Đã tải Hybrid CF model')
except Exception as e:
    # Nếu custom unpickler không work, thử load bình thường
    print(f'Thử load lại với phương pháp khác...')
    with open(CF_MODEL_PATH, 'rb') as f:
        cf_data = pickle.load(f)
    print('✅ Đã tải Hybrid CF model')

cbf_models = cbf_data['models']          # dict: user_id -> ElasticNet/Ridge
X_items = cbf_data['X']                   # item feature matrix
item2idx = cbf_data['item2idx']
idx2item = cbf_data['idx2item']
cbf_global_mean = np.mean(X_items)

cf_model = cf_data['model']               # HybridCF model
user_map = cf_data['user_map']
activity_map = cf_data['activity_map']
inv_activity_map = {v: k for k, v in activity_map.items()}

# ============================================================
# 2. PREDICTION FUNCTIONS
# ============================================================

def predict_cbf(user_id, item_idx):
    """Predict rating by Content-Based Filtering"""
    if user_id in cbf_models:
        return cbf_models[user_id].predict(X_items[item_idx].reshape(1, -1))[0]
    return cbf_global_mean


def predict_cf(user_id, activity_id):
    """Predict rating by Collaborative Filtering
    user_id: original user_id từ ratings.csv
    activity_id: original activity_id từ items.csv
    """
    # Kiểm tra xem user_id và activity_id có trong mapping không
    if user_id not in user_map or activity_id not in activity_map:
        return 3.0
    
    # Map về index trong model
    mapped_user = user_map[user_id]
    mapped_activity = activity_map[activity_id]
    
    try:
        return cf_model.pred(int(mapped_user), int(mapped_activity))
    except:
        return 3.0


def predict_hybrid(user_id, activity_id, beta=0.5):
    """Final hybrid prediction
    user_id: original user_id từ ratings.csv
    activity_id: original activity_id từ items.csv
    beta: weight for CF, (1-beta) for CBF
    """
    # Lấy item_idx từ activity_id
    if activity_id not in item2idx:
        return 3.0
    
    item_idx = item2idx[activity_id]
    
    # Predict từ cả 2 models
    r_cf = predict_cf(user_id, activity_id)
    r_cbf = predict_cbf(user_id, item_idx)
    
    # Kết hợp và clip về range [1, 5]
    final_pred = beta * r_cf + (1 - beta) * r_cbf
    return np.clip(final_pred, 1, 5)


# ============================================================
# 3. TOP-N RECOMMENDATION
# ============================================================

def recommend_top_n(user_id, rated_activity_ids, n=5, beta=0.5):
    """Recommend top N activities for a user
    user_id: original user_id từ ratings.csv
    rated_activity_ids: set of activity_ids mà user đã rate
    """
    scores = []
    for activity_id in idx2item.values():
        if activity_id in rated_activity_ids:
            continue
        score = predict_hybrid(user_id, activity_id, beta)
        scores.append((activity_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


# ============================================================
# 4. DEMO
# ============================================================

if __name__ == '__main__':
    # Load dữ liệu để lấy danh sách users và activities
    print('\nĐang tải dữ liệu...')
    ratings = pd.read_csv('../data/ratings.csv')
    items = pd.read_csv('../data/items.csv')
    
    # Lấy danh sách unique users và activities
    unique_users = sorted(ratings['user_id'].unique())
    unique_activities = sorted(items['item_id'].unique())
    
    print(f'Số lượng users: {len(unique_users)}')
    print(f'Số lượng activities: {len(unique_activities)}')
    
    # Tạo recommendations cho tất cả user-activity pairs
    print('\nĐang tạo recommendations...')
    recommendations = []
    
    for user_id in unique_users:
        # Lấy các activities mà user đã rate
        user_rated = set(ratings[ratings['user_id'] == user_id]['activity_id'].unique())
        
        for activity_id in unique_activities:
            # Bỏ qua các activities đã rate
            if activity_id in user_rated:
                continue
            
            pred_rating = predict_hybrid(user_id, activity_id, beta=0.6)
            recommendations.append({
                'user_id': user_id,
                'activity_id': activity_id,
                'predicted_rating': round(pred_rating, 4)
            })
    
    # Lưu recommendations
    recommendations_df = pd.DataFrame(recommendations)
    output_path = '../data/recommendations.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    recommendations_df.to_csv(output_path, index=False)
    
    print(f'\n✅ Đã lưu recommendations vào: {output_path}')
    print(f'   Tổng số recommendations: {len(recommendations_df):,}')
    print(f'   Rating trung bình: {recommendations_df["predicted_rating"].mean():.4f}')
    print(f'   Rating min: {recommendations_df["predicted_rating"].min():.4f}')
    print(f'   Rating max: {recommendations_df["predicted_rating"].max():.4f}')
    
    # Hiển thị một vài recommendations mẫu
    print('\n📊 Một vài recommendations mẫu:')
    print(recommendations_df.head(10).to_string(index=False))
    
    print('\n' + '=' * 60)
    print('HOÀN TẤT!')
    print('=' * 60)