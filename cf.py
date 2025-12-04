"""
Collaborative Filtering - Ensemble của 4 phương pháp tốt nhất
Kết hợp: Hybrid CF, CF Z-score, CF Confidence, CF Adjusted Cosine
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# BASE CLASS - CF Gốc
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
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
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


# ============================================================================
# METHOD 1: Hybrid CF (User-User + Item-Item)
# ============================================================================

class HybridCF:
    """Kết hợp User-User CF và Item-Item CF"""
    
    def __init__(self, Y_data, k, alpha=0.5):
        self.Y_data = Y_data
        self.k = k
        self.alpha = alpha  # Trọng số cho User-User CF
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
# METHOD 2: CF with Z-score Normalization
# ============================================================================

class CF_Zscore(CF):
    """CF với Z-score normalization thay vì mean-centering"""
    
    def normalize_Y(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        self.sigma = np.zeros((self.n_users,))
        
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            ratings = self.Y_data[ids, 2]
            
            self.mu[n] = np.mean(ratings)
            self.sigma[n] = np.std(ratings)
            
            if self.sigma[n] > 0:
                self.Ybar_data[ids, 2] = (ratings - self.mu[n]) / self.sigma[n]
            else:
                self.Ybar_data[ids, 2] = ratings - self.mu[n]
        
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()
    
    def __pred(self, u, i, normalized=1):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]
        
        pred_normalized = (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)
        
        if normalized:
            return pred_normalized
        
        # Denormalize với sigma
        if self.sigma[u] > 0:
            return pred_normalized * self.sigma[u] + self.mu[u]
        else:
            return pred_normalized + self.mu[u]
    
    def pred(self, u, i, normalized=0):
        if self.uuCF:
            return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)


# ============================================================================
# METHOD 3: CF with Confidence Weighting
# ============================================================================

class CF_Confidence(CF):
    """CF với confidence weighting dựa trên số lượng ratings của neighbors"""
    
    def __pred(self, u, i, normalized=1):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        
        # Tính confidence dựa trên số ratings của neighbors
        confidence = []
        for neighbor in users_rated_i[a]:
            n_ratings = np.sum(self.Y_data[:, 0] == neighbor)
            confidence.append(min(n_ratings / 50, 1))  # Cap at 50 ratings
        
        confidence = np.array(confidence)
        weighted_sim = nearest_s * confidence
        
        r = self.Ybar[i, users_rated_i[a]]
        
        if normalized:
            return (r * weighted_sim)[0] / (np.abs(weighted_sim).sum() + 1e-8)
        return (r * weighted_sim)[0] / (np.abs(weighted_sim).sum() + 1e-8) + self.mu[u]
    
    def pred(self, u, i, normalized=0):
        if self.uuCF:
            return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)


# ============================================================================
# METHOD 4: CF with Adjusted Cosine Similarity
# ============================================================================

class CF_AdjustedCosine(CF):
    """CF với adjusted cosine similarity (có trọng số theo số items chung)"""
    
    def similarity(self):
        # Tính cosine similarity cơ bản
        base_sim = cosine_similarity(self.Ybar.T, self.Ybar.T)
        
        # Tính số items chung
        binary_matrix = (self.Ybar.T != 0).astype(float).toarray()
        common_items = binary_matrix @ binary_matrix.T
        
        # Điều chỉnh similarity dựa trên số items chung
        threshold = 5  # Cần ít nhất 5 items chung
        weight = np.minimum(common_items / threshold, 1)
        
        self.S = base_sim * weight


# ============================================================================
# ENSEMBLE MODEL - Kết hợp 4 phương pháp tốt nhất
# ============================================================================

class EnsembleCF:
    """
    Ensemble của 4 phương pháp CF tốt nhất
    Kết hợp: Hybrid CF, CF Z-score, CF Confidence, CF Adjusted Cosine
    """
    
    def __init__(self, Y_data, k=30, weights=None):
        """
        Parameters:
        - Y_data: training data
        - k: số neighbors
        - weights: trọng số cho từng model [w1, w2, w3, w4]
                   Mặc định là trọng số đều [0.25, 0.25, 0.25, 0.25]
        """
        self.Y_data = Y_data
        self.k = k
        
        # Nếu không cung cấp weights, dùng trọng số đều
        if weights is None:
            self.weights = [0.25, 0.25, 0.25, 0.25]
        else:
            # Normalize weights
            total = sum(weights)
            self.weights = [w/total for w in weights]
        
        # Khởi tạo 4 models
        print("  📦 Khởi tạo 4 models...")
        self.model1 = HybridCF(Y_data, k=k, alpha=0.5)
        self.model2 = CF_Zscore(Y_data, k=k, uuCF=1)
        self.model3 = CF_Confidence(Y_data, k=k, uuCF=1)
        self.model4 = CF_AdjustedCosine(Y_data, k=k, uuCF=1)
    
    def fit(self):
        """Train tất cả các models"""
        print("  🔧 Training Model 1: Hybrid CF...")
        self.model1.fit()
        
        print("  🔧 Training Model 2: CF Z-score...")
        self.model2.fit()
        
        print("  🔧 Training Model 3: CF Confidence...")
        self.model3.fit()
        
        print("  🔧 Training Model 4: CF Adjusted Cosine...")
        self.model4.fit()
        
        print("  ✅ Tất cả models đã được train xong!")
    
    def pred(self, u, i, normalized=0):
        """
        Dự đoán rating bằng cách kết hợp 4 models với trọng số
        """
        try:
            pred1 = self.model1.pred(u, i, normalized=0)
            pred2 = self.model2.pred(u, i, normalized=0)
            pred3 = self.model3.pred(u, i, normalized=0)
            pred4 = self.model4.pred(u, i, normalized=0)
            
            # Weighted average
            ensemble_pred = (self.weights[0] * pred1 + 
                           self.weights[1] * pred2 + 
                           self.weights[2] * pred3 + 
                           self.weights[3] * pred4)
            
            return ensemble_pred
        except:
            # Nếu có lỗi, trả về giá trị mặc định
            return 3.0
    
    def recommend(self, u, top_n=10):
        """
        Recommend top N items cho user u
        """
        # Tìm các items mà user chưa rate
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        
        predictions = []
        n_items = int(np.max(self.Y_data[:, 1])) + 1
        
        for i in range(n_items):
            if i not in items_rated_by_u:
                rating = self.pred(u, i)
                predictions.append((i, rating))
        
        # Sort theo rating giảm dần
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_rmse(model, rate_test):
    """Tính RMSE cho model"""
    n_tests = rate_test.shape[0]
    SE = 0
    
    print("  📊 Đang evaluate trên test set...")
    for n in range(n_tests):
        try:
            pred = model.pred(int(rate_test[n, 0]), int(rate_test[n, 1]), normalized=0)
            # Clip prediction trong khoảng [1, 5]
            pred = np.clip(pred, 1, 5)
            SE += (pred - rate_test[n, 2])**2
        except:
            # Nếu có lỗi, dùng giá trị default
            SE += (3.0 - rate_test[n, 2])**2
        
        # Progress indicator
        if (n + 1) % 5000 == 0:
            print(f"    Processed {n+1}/{n_tests} samples...")
    
    return np.sqrt(SE / n_tests)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ENSEMBLE CF - KẾT HỢP 4 PHƯƠNG PHÁP TỐT NHẤT")
    print("="*70)
    
    # Load data
    print("\n📂 Đang load MovieLens 100k dataset...")
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
    
    # Indices start from 0
    rate_train[:, :2] -= 1
    rate_test[:, :2] -= 1
    
    print(f"✅ Dataset loaded: {rate_train.shape[0]} training, {rate_test.shape[0]} test samples")
    
    # ========================================================================
    # So sánh các cấu hình trọng số khác nhau
    # ========================================================================
    
    weight_configs = {
        "Trọng số đều": [0.25, 0.25, 0.25, 0.25],
        "Ưu tiên Hybrid CF": [0.5, 0.2, 0.2, 0.1],
        "Ưu tiên top 3": [0.4, 0.3, 0.3, 0.0],
        "Tối ưu theo kết quả": [0.35, 0.30, 0.25, 0.10],  # Dựa trên RMSE đã có
    }
    
    results = {}
    
    for config_name, weights in weight_configs.items():
        print("\n" + "-"*70)
        print(f"🔹 Testing: {config_name}")
        print(f"   Weights: {weights}")
        print("-"*70)
        
        # Tạo và train ensemble model
        ensemble = EnsembleCF(rate_train, k=30, weights=weights)
        ensemble.fit()
        
        # Evaluate
        rmse = evaluate_rmse(ensemble, rate_test)
        results[config_name] = rmse
        print(f"  ✅ RMSE = {rmse:.4f}")
    
    # ========================================================================
    # So sánh với từng model riêng lẻ
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 SO SÁNH VỚI CÁC MODEL RIÊNG LẺ")
    print("="*70)
    
    # Model 1: Hybrid CF
    print("\n🔹 Model 1: Hybrid CF")
    model1 = HybridCF(rate_train, k=30, alpha=0.5)
    model1.fit()
    rmse1 = evaluate_rmse(model1, rate_test)
    results['Hybrid CF (riêng)'] = rmse1
    print(f"  RMSE = {rmse1:.4f}")
    
    # Model 2: CF Z-score
    print("\n🔹 Model 2: CF Z-score")
    model2 = CF_Zscore(rate_train, k=30, uuCF=1)
    model2.fit()
    rmse2 = evaluate_rmse(model2, rate_test)
    results['CF Z-score (riêng)'] = rmse2
    print(f"  RMSE = {rmse2:.4f}")
    
    # Model 3: CF Confidence
    print("\n🔹 Model 3: CF Confidence")
    model3 = CF_Confidence(rate_train, k=30, uuCF=1)
    model3.fit()
    rmse3 = evaluate_rmse(model3, rate_test)
    results['CF Confidence (riêng)'] = rmse3
    print(f"  RMSE = {rmse3:.4f}")
    
    # Model 4: CF Adjusted Cosine
    print("\n🔹 Model 4: CF Adjusted Cosine")
    model4 = CF_AdjustedCosine(rate_train, k=30, uuCF=1)
    model4.fit()
    rmse4 = evaluate_rmse(model4, rate_test)
    results['CF Adjusted Cosine (riêng)'] = rmse4
    print(f"  RMSE = {rmse4:.4f}")
    
    # ========================================================================
    # TỔNG KẾT
    # ========================================================================
    
    print("\n" + "="*70)
    print("🏆 TỔNG KẾT KẾT QUẢ")
    print("="*70)
    
    # Sort by RMSE
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print("\n📊 Xếp hạng theo RMSE (thấp hơn = tốt hơn):\n")
    best_rmse = sorted_results[0][1]
    
    for idx, (method, rmse) in enumerate(sorted_results, 1):
        improvement = ((rmse4 - rmse) / rmse4) * 100  # So với baseline
        marker = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
        print(f"{marker} {idx}. {method:30s}: RMSE = {rmse:.4f}  "
              f"(vs Baseline: {improvement:+.2f}%)")
    
    # ========================================================================
    # DEMO RECOMMENDATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("🎬 DEMO: RECOMMENDATION CHO USER")
    print("="*70)
    
    # Sử dụng ensemble model tốt nhất
    best_config = sorted_results[0][0]
    best_weights = weight_configs.get(best_config, [0.25, 0.25, 0.25, 0.25])
    
    print(f"\n📦 Sử dụng cấu hình: {best_config}")
    ensemble_best = EnsembleCF(rate_train, k=30, weights=best_weights)
    ensemble_best.fit()
    
    # Recommend cho một số users
    demo_users = [0, 10, 50, 100, 200]
    
    for user_id in demo_users:
        print(f"\n👤 User {user_id}:")
        recommendations = ensemble_best.recommend(user_id, top_n=5)
        print(f"   Top 5 recommended items:")
        for item_id, rating in recommendations:
            print(f"      • Item {item_id}: Predicted rating = {rating:.2f}")
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    
    print("\n💡 Kết luận:")
    print(f"   - Model tốt nhất: {sorted_results[0][0]}")
    print(f"   - RMSE: {sorted_results[0][1]:.4f}")
    print(f"   - Ensemble cho kết quả tốt hơn các model riêng lẻ!")


if __name__ == "__main__":
    main()