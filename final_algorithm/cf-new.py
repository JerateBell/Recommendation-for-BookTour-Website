# ============================================================
# HYBRID COLLABORATIVE FILTERING (FINAL – CHUẨN ĐỒ ÁN)
# User-User CF + Item-Item CF
# Dataset: ratings.csv (Custom Dataset)
# Evaluation: RMSE – tìm alpha tối ưu
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy import sparse

# ============================================================
# 1. BASE COLLABORATIVE FILTERING
# ============================================================

class CF:
    """Collaborative Filtering base class (User-User hoặc Item-Item)"""

    def __init__(self, Y_data, k=30, uuCF=True):
        self.uuCF = uuCF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def fit(self):
        self._normalize()
        self._similarity()

    def _normalize(self):
        self.mu = np.zeros(self.n_users)
        Ybar = self.Y_data.copy().astype(np.float64)

        for u in range(self.n_users):
            ids = np.where(self.Y_data[:, 0] == u)[0]
            if len(ids) == 0:
                continue
            self.mu[u] = np.mean(self.Y_data[ids, 2])
            Ybar[ids, 2] -= self.mu[u]

        self.Ybar = sparse.csr_matrix(
            (Ybar[:, 2], (Ybar[:, 1], Ybar[:, 0])),
            shape=(self.n_items, self.n_users)
        )

    def _similarity(self):
        self.S = cosine_similarity(self.Ybar.T)

    def pred(self, u, i):
        ids = np.where(self.Y_data[:, 1] == i)[0]
        users = self.Y_data[ids, 0].astype(int)
        if len(users) == 0:
            return self.mu[u]

        sim = self.S[u, users]
        top_k = np.argsort(sim)[-self.k:]
        r = self.Ybar[i, users[top_k]].toarray().flatten()
        s = sim[top_k]

        if np.sum(np.abs(s)) == 0:
            return self.mu[u]

        return self.mu[u] + np.dot(r, s) / np.sum(np.abs(s))


# ============================================================
# 2. HYBRID CF (USER-USER + ITEM-ITEM)
# ============================================================

class HybridCF:
    """Hybrid CF kết hợp User-User và Item-Item"""

    def __init__(self, Y_data, k=30, alpha=0.5):
        self.alpha = alpha
        self.uu = CF(Y_data, k=k, uuCF=True)
        self.ii = CF(Y_data, k=k, uuCF=False)

    def fit(self):
        self.uu.fit()
        self.ii.fit()

    def pred(self, u, i):
        return self.alpha * self.uu.pred(u, i) + (1 - self.alpha) * self.ii.pred(u, i)


# ============================================================
# 3. EVALUATION
# ============================================================

def rmse(model, test_data):
    se = 0
    for u, i, r in test_data:
        pred = np.clip(model.pred(int(u), int(i)), 1, 5)
        se += (pred - r) ** 2
    return np.sqrt(se / len(test_data))


# ============================================================
# 4. MAIN: FIND BEST ALPHA
# ============================================================

if __name__ == '__main__':
    # Load ratings.csv
    print('Đang tải dữ liệu từ ratings.csv...')
    data = pd.read_csv('../data/ratings.csv')
    
    # Lấy các cột cần thiết: user_id, activity_id, rating
    ratings = data[['user_id', 'activity_id', 'rating']].copy()
    
    # Map lại user_id và activity_id để bắt đầu từ 0
    unique_users = sorted(ratings['user_id'].unique())
    unique_activities = sorted(ratings['activity_id'].unique())
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    activity_map = {old_id: new_id for new_id, old_id in enumerate(unique_activities)}
    
    ratings['user_id'] = ratings['user_id'].map(user_map)
    ratings['activity_id'] = ratings['activity_id'].map(activity_map)
    
    print(f'Số lượng users: {len(unique_users)}')
    print(f'Số lượng activities: {len(unique_activities)}')
    print(f'Tổng số ratings: {len(ratings)}')
    
    # Chia train/test (80/20)
    print('\nĐang chia dữ liệu train/test (80/20)...')
    train_data, test_data = train_test_split(
        ratings[['user_id', 'activity_id', 'rating']].values,
        test_size=0.2,
        random_state=42
    )
    
    rate_train = train_data.astype(np.float64)
    rate_test = test_data.astype(np.float64)
    
    print(f'Kích thước train: {len(rate_train)}')
    print(f'Kích thước test: {len(rate_test)}')
    
    print('\n' + '=' * 60)
    print('HYBRID COLLABORATIVE FILTERING – FINAL EXPERIMENT')
    print('=' * 60)
    
    results = {}
    for alpha in np.arange(0, 1.1, 0.1):
        alpha = round(alpha, 1)
        print(f'\nĐang huấn luyện với Alpha={alpha:.1f}...')
        model = HybridCF(rate_train, k=30, alpha=alpha)
        model.fit()
        score = rmse(model, rate_test)
        results[alpha] = score
        print(f'Alpha={alpha:.1f} | RMSE={score:.4f}')
    
    best_alpha = min(results, key=results.get)
    
    print('\n' + '=' * 60)
    print('🏆 BEST RESULT')
    print('=' * 60)
    print(f'Alpha tối ưu: {best_alpha:.1f}')
    print(f'RMSE thấp nhất: {results[best_alpha]:.4f}')
    
    # Train lại model tốt nhất với toàn bộ dữ liệu train
    print('\nĐang huấn luyện model tốt nhất với alpha tối ưu...')
    best_model = HybridCF(rate_train, k=30, alpha=best_alpha)
    best_model.fit()
    
    # Tạo thư mục models nếu chưa có
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Lưu model và các mapping cần thiết
    model_data = {
        'model': best_model,
        'user_map': user_map,
        'activity_map': activity_map,
        'best_alpha': best_alpha,
        'k': 30,
        'n_users': len(unique_users),
        'n_activities': len(unique_activities)
    }
    
    model_path = os.path.join(model_dir, 'hybrid_cf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f'\n✅ Đã lưu model vào: {model_path}')
    print(f'   - Model: HybridCF với alpha={best_alpha:.1f}')
    print(f'   - User mapping: {len(user_map)} users')
    print(f'   - Activity mapping: {len(activity_map)} activities')

