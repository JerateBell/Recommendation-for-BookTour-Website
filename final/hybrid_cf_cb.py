"""
Hybrid Recommendation System
Kết hợp Collaborative Filtering (CF) và Content-Based Filtering (CB)

Phương pháp: Weighted Ensemble
- CF: Dựa trên similarity giữa users/items
- CB: Dựa trên features của items (thể loại phim)
- Kết hợp: weighted average của 2 predictions
"""

import sys
import io
# Fix encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import ElasticNet, Ridge
from scipy import sparse
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# COLLABORATIVE FILTERING (từ cf.py)
# ============================================================================

class CF(object):
    """Base Collaborative Filtering class"""
    
    def __init__(self, Y_data, k, dist_func=cosine_similarity, uuCF=1):
        self.uuCF = uuCF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
    
    def normalize_Y(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            ratings = self.Y_data[ids, 2]
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0
            self.mu[n] = m
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Y_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
    
    def fit(self):
        self.normalize_Y()
        self.similarity()
    
    def __pred(self, u, i, normalized=1):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)
        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized=1):
        if self.uuCF:
            return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)


class HybridCF:
    """Kết hợp User-User CF và Item-Item CF"""
    
    def __init__(self, Y_data, k, alpha=0.5):
        self.Y_data = Y_data
        self.k = k
        self.alpha = alpha
        self.uu_cf = CF(Y_data, k, uuCF=1)
        self.ii_cf = CF(Y_data, k, uuCF=0)
    
    def fit(self):
        self.uu_cf.fit()
        self.ii_cf.fit()
    
    def pred(self, u, i, normalized=0):
        pred_uu = self.uu_cf.pred(u, i, normalized=0)
        pred_ii = self.ii_cf.pred(u, i, normalized=0)
        return self.alpha * pred_uu + (1 - self.alpha) * pred_ii


# ============================================================================
# CONTENT-BASED FILTERING (từ content-based.py)
# ============================================================================

class ContentBasedFiltering:
    """Content-Based Filtering với ElasticNet"""
    
    def __init__(self, rate_train, tfidf, alpha=0.01, l1_ratio=0.5):
        self.rate_train = rate_train
        self.tfidf = tfidf
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.W = None
        self.b = None
        self.n_users = int(np.max(rate_train[:, 0])) + 1
        self.d = tfidf.shape[1]
    
    def get_items_rated_by_user(self, rate_matrix, user_id):
        """return (item_ids, scores)"""
        y = rate_matrix[:, 0]
        ids = np.where(y == user_id + 1)[0]
        item_ids = rate_matrix[ids, 1] - 1
        scores = rate_matrix[ids, 2]
        return (item_ids, scores)
    
    def fit(self):
        """Train ElasticNet model cho từng user"""
        global_mean = np.mean(self.rate_train[:, 2])
        MIN_RATINGS = 5
        
        self.W = np.zeros((self.d, self.n_users))
        self.b = np.zeros((1, self.n_users))
        
        print(f"  🔧 Training Content-Based models cho {self.n_users} users...")
        for n in range(self.n_users):
            ids, scores = self.get_items_rated_by_user(self.rate_train, n)
            
            if len(ids) < MIN_RATINGS:
                self.b[0, n] = global_mean
                self.W[:, n] = 0
                continue
            
            Xhat = self.tfidf[ids, :]
            clf = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, 
                            fit_intercept=True, max_iter=1000)
            try:
                clf.fit(Xhat, scores)
                self.W[:, n] = clf.coef_
                self.b[0, n] = clf.intercept_
            except:
                clf_fallback = Ridge(alpha=0.1, fit_intercept=True)
                clf_fallback.fit(Xhat, scores)
                self.W[:, n] = clf_fallback.coef_
                self.b[0, n] = clf_fallback.intercept_
            
            if (n + 1) % 100 == 0:
                print(f"    Processed {n+1}/{self.n_users} users...")
        
        print("  ✅ Content-Based training completed!")
    
    def pred(self, u, i):
        """Dự đoán rating của user u cho item i"""
        if self.W is None or self.b is None:
            raise ValueError("Model chưa được train. Gọi fit() trước.")
        
        # Yhat[i, u] = tfidf[i, :] × W[:, u] + b[u]
        pred = self.tfidf[i, :].dot(self.W[:, u]) + self.b[0, u]
        return pred


# ============================================================================
# HYBRID SYSTEM: CF + Content-Based
# ============================================================================

class HybridCFCB:
    """
    Kết hợp Collaborative Filtering và Content-Based Filtering
    """
    
    def __init__(self, rate_train, tfidf, cf_k=30, cf_alpha=0.5, 
                 cb_alpha=0.01, cb_l1_ratio=0.5, weight_cf=0.5):
        """
        Parameters:
        - rate_train: training ratings data
        - tfidf: TF-IDF matrix của items
        - cf_k: số neighbors cho CF
        - cf_alpha: trọng số User-User vs Item-Item CF
        - cb_alpha: alpha cho ElasticNet trong Content-Based
        - cb_l1_ratio: l1_ratio cho ElasticNet
        - weight_cf: trọng số cho CF (weight_cb = 1 - weight_cf)
        """
        self.rate_train = rate_train
        self.tfidf = tfidf
        self.weight_cf = weight_cf
        self.weight_cb = 1 - weight_cf
        
        # Khởi tạo 2 models
        print("📦 Khởi tạo models...")
        print(f"  - Collaborative Filtering (k={cf_k}, alpha={cf_alpha})")
        print(f"  - Content-Based Filtering (alpha={cb_alpha}, l1_ratio={cb_l1_ratio})")
        print(f"  - Ensemble weights: CF={weight_cf:.2f}, CB={self.weight_cb:.2f}")
        
        self.cf_model = HybridCF(rate_train, k=cf_k, alpha=cf_alpha)
        self.cb_model = ContentBasedFiltering(rate_train, tfidf, 
                                             alpha=cb_alpha, l1_ratio=cb_l1_ratio)
    
    def fit(self):
        """Train cả 2 models"""
        print("\n🔧 Training Collaborative Filtering...")
        self.cf_model.fit()
        
        print("\n🔧 Training Content-Based Filtering...")
        self.cb_model.fit()
        
        print("\n✅ Tất cả models đã được train xong!")
    
    def pred(self, u, i):
        """
        Dự đoán rating bằng cách kết hợp CF và Content-Based
        """
        try:
            # Prediction từ CF
            pred_cf = self.cf_model.pred(u, i, normalized=0)
            pred_cf = np.clip(pred_cf, 1, 5)  # Clip trong [1, 5]
        except:
            pred_cf = 3.0  # Default value
        
        try:
            # Prediction từ Content-Based
            pred_cb = self.cb_model.pred(u, i)
            pred_cb = np.clip(pred_cb, 1, 5)  # Clip trong [1, 5]
        except:
            pred_cb = 3.0  # Default value
        
        # Weighted ensemble
        pred_hybrid = self.weight_cf * pred_cf + self.weight_cb * pred_cb
        
        return pred_hybrid
    
    def recommend(self, u, top_n=10):
        """Recommend top N items cho user u"""
        # Tìm các items mà user chưa rate
        ids = np.where(self.rate_train[:, 0] == u)[0]
        items_rated_by_u = self.rate_train[ids, 1].tolist()
        
        predictions = []
        n_items = int(np.max(self.rate_train[:, 1])) + 1
        
        for i in range(n_items):
            if i not in items_rated_by_u:
                rating = self.pred(u, i)
                predictions.append((i, rating))
        
        # Sort theo rating giảm dần
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_rmse(model, rate_test):
    """Tính RMSE cho model"""
    n_tests = rate_test.shape[0]
    SE = 0
    
    print("  📊 Đang evaluate trên test set...")
    for n in range(n_tests):
        try:
            pred = model.pred(int(rate_test[n, 0]), int(rate_test[n, 1]))
            pred = np.clip(pred, 1, 5)
            SE += (pred - rate_test[n, 2])**2
        except:
            SE += (3.0 - rate_test[n, 2])**2
        
        if (n + 1) % 5000 == 0:
            print(f"    Processed {n+1}/{n_tests} samples...")
    
    return np.sqrt(SE / n_tests)


def evaluate_with_accuracy(model, rate_test, n_users):
    """Tính RMSE và Accuracy"""
    se = 0
    cnt = 0
    within_1 = 0
    
    for n in range(n_users):
        ids = np.where(rate_test[:, 0] == n + 1)[0]
        if len(ids) == 0:
            continue
        
        for idx in ids:
            u = int(rate_test[idx, 0]) - 1
            i = int(rate_test[idx, 1]) - 1
            true_rating = rate_test[idx, 2]
            
            try:
                pred = model.pred(u, i)
                pred = np.clip(pred, 1, 5)
                e = true_rating - pred
                se += e**2
                cnt += 1
                if abs(e) <= 1.0:
                    within_1 += 1
            except:
                se += (3.0 - true_rating)**2
                cnt += 1
    
    rmse = sqrt(se/cnt) if cnt > 0 else float('inf')
    accuracy = (within_1 / cnt * 100) if cnt > 0 else 0
    
    return rmse, accuracy


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("HYBRID RECOMMENDATION SYSTEM")
    print("Kết hợp Collaborative Filtering + Content-Based Filtering")
    print("="*70)
    
    # ========================================================================
    # Load data
    # ========================================================================
    print("\n📂 Đang load MovieLens 100k dataset...")
    
    # Ratings data (cho CF)
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    try:
        ratings_base = pd.read_csv('ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
        ratings_test = pd.read_csv('ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')
    except:
        print("❌ Không tìm thấy file dataset!")
        print("   Vui lòng tải MovieLens 100k và đặt trong folder 'ml-100k/'")
        return
    
    rate_train = ratings_base.to_numpy()
    rate_test = ratings_test.to_numpy()
    rate_train[:, :2] -= 1
    rate_test[:, :2] -= 1
    
    print(f"✅ Ratings loaded: {rate_train.shape[0]} training, {rate_test.shape[0]} test samples")
    
    # Items data (cho Content-Based)
    i_cols = [
        'movie id', 'movie title', 'release date', 'video release date', 'IMDb URL',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
        'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
    X0 = items.to_numpy()
    X_train_counts = X0[:, -19:]  # 19 thể loại phim
    
    # TF-IDF
    transformer = TfidfTransformer(smooth_idf=True, norm='l2')
    tfidf = transformer.fit_transform(X_train_counts).toarray()
    
    print(f"✅ Items loaded: {items.shape[0]} items, TF-IDF shape: {tfidf.shape}")
    
    # ========================================================================
    # Test các trọng số khác nhau
    # ========================================================================
    print("\n" + "="*70)
    print("🔍 TÌM TRỌNG SỐ TỐI ƯU CHO ENSEMBLE")
    print("="*70)
    
    weight_configs = [
        (0.0, "Chỉ Content-Based"),
        (0.25, "CB 75%, CF 25%"),
        (0.5, "CB 50%, CF 50%"),
        (0.75, "CB 25%, CF 75%"),
        (1.0, "Chỉ CF")
    ]
    
    results = {}
    
    for weight_cf, config_name in weight_configs:
        print("\n" + "-"*70)
        print(f"🔹 Testing: {config_name}")
        print(f"   Weight CF = {weight_cf:.2f}, Weight CB = {1-weight_cf:.2f}")
        print("-"*70)
        
        # Tạo và train hybrid model
        hybrid_model = HybridCFCB(rate_train, tfidf, 
                                 cf_k=30, cf_alpha=0.5,
                                 cb_alpha=0.01, cb_l1_ratio=0.5,
                                 weight_cf=weight_cf)
        hybrid_model.fit()
        
        # Evaluate
        rmse = evaluate_rmse(hybrid_model, rate_test)
        results[config_name] = (weight_cf, rmse)
        print(f"  ✅ RMSE = {rmse:.4f}\n")
    
    # ========================================================================
    # TỔNG KẾT
    # ========================================================================
    print("\n" + "="*70)
    print("🏆 TỔNG KẾT KẾT QUẢ")
    print("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][1])
    best_config, (best_weight_cf, best_rmse) = sorted_results[0]
    
    print("\n📊 Xếp hạng theo RMSE (thấp hơn = tốt hơn):\n")
    
    for idx, (config_name, (weight_cf, rmse)) in enumerate(sorted_results, 1):
        improvement = ((rmse - best_rmse) / best_rmse) * 100
        marker = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
        print(f"{marker} {idx}. {config_name:25s}  "
              f"(CF: {weight_cf*100:.0f}%, CB: {(1-weight_cf)*100:.0f}%)  "
              f": RMSE = {rmse:.4f}  "
              f"(chênh lệch: {improvement:+.2f}%)")
    
    # ========================================================================
    # DEMO RECOMMENDATION
    # ========================================================================
    print("\n" + "="*70)
    print("🎬 DEMO: RECOMMENDATION CHO USER (VỚI TRỌNG SỐ TỐI ƯU)")
    print("="*70)
    
    print(f"\n📦 Sử dụng cấu hình tốt nhất: {best_config}")
    print(f"   Weight CF = {best_weight_cf:.2f}, Weight CB = {1-best_weight_cf:.2f}")
    
    # Train lại với best weights
    print("\n  🔧 Training model với trọng số tối ưu...")
    model_best = HybridCFCB(rate_train, tfidf,
                            cf_k=30, cf_alpha=0.5,
                            cb_alpha=0.01, cb_l1_ratio=0.5,
                            weight_cf=best_weight_cf)
    model_best.fit()
    
    # Recommend cho một số users
    demo_users = [0, 10, 50, 100, 200]
    
    for user_id in demo_users:
        print(f"\n👤 User {user_id}:")
        recommendations = model_best.recommend(user_id, top_n=5)
        print(f"   Top 5 recommended items:")
        for item_id, rating in recommendations:
            print(f"      • Item {item_id}: Predicted rating = {rating:.2f}")
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    
    print("\n💡 Kết luận:")
    print(f"   - Cấu hình tốt nhất: {best_config}")
    print(f"   - Weight CF: {best_weight_cf:.2f}")
    print(f"   - Weight CB: {1-best_weight_cf:.2f}")
    print(f"   - RMSE tốt nhất: {best_rmse:.4f}")


if __name__ == "__main__":
    main()

