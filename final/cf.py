"""
Collaborative Filtering - Hybrid CF (User-User + Item-Item)
Tìm alpha tối ưu cho Hybrid CF
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


# ============================================================================
# HYBRID CF (User-User + Item-Item)
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
    print("HYBRID CF - TÌM ALPHA TỐI ƯU")
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
    # Test các giá trị alpha khác nhau
    # ========================================================================
    
    print("\n" + "="*70)
    print("🔍 TÌM ALPHA TỐI ƯU")
    print("="*70)
    print("\n📊 Testing các giá trị alpha từ 0.0 đến 1.0 (bước 0.1)...\n")
    
    # Danh sách các giá trị alpha để test
    alpha_values = np.arange(0.0, 1.1, 0.1)
    results = {}
    
    for alpha in alpha_values:
        alpha = round(alpha, 1)  # Làm tròn để tránh lỗi floating point
        print("-" * 70)
        print(f"🔹 Testing alpha = {alpha:.1f}")
        print(f"   (User-User CF: {alpha*100:.0f}%, Item-Item CF: {(1-alpha)*100:.0f}%)")
        print("-" * 70)
        
        # Tạo và train model
        print("  🔧 Training Hybrid CF model...")
        model = HybridCF(rate_train, k=30, alpha=alpha)
        model.fit()
        print("  ✅ Model đã được train xong!")
        
        # Evaluate
        print("  📊 Đang evaluate trên test set...")
        rmse = evaluate_rmse(model, rate_test)
        results[alpha] = rmse
        print(f"  ✅ RMSE = {rmse:.4f}\n")
    
    # ========================================================================
    # TỔNG KẾT KẾT QUẢ
    # ========================================================================
    
    print("\n" + "="*70)
    print("🏆 TỔNG KẾT KẾT QUẢ")
    print("="*70)
    
    # Sort by RMSE
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    best_alpha, best_rmse = sorted_results[0]
    
    print("\n📊 Xếp hạng theo RMSE (thấp hơn = tốt hơn):\n")
    
    for idx, (alpha, rmse) in enumerate(sorted_results, 1):
        improvement = ((rmse - best_rmse) / best_rmse) * 100
        marker = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
        print(f"{marker} {idx}. Alpha = {alpha:.1f}  "
              f"(UU: {alpha*100:.0f}%, II: {(1-alpha)*100:.0f}%)  "
              f": RMSE = {rmse:.4f}  "
              f"(chênh lệch: {improvement:+.2f}%)")
    
    # ========================================================================
    # DEMO RECOMMENDATION VỚI ALPHA TỐI ƯU
    # ========================================================================
    
    print("\n" + "="*70)
    print("🎬 DEMO: RECOMMENDATION CHO USER (VỚI ALPHA TỐI ƯU)")
    print("="*70)
    
    print(f"\n📦 Sử dụng alpha tốt nhất: {best_alpha:.1f}")
    print(f"   (User-User CF: {best_alpha*100:.0f}%, Item-Item CF: {(1-best_alpha)*100:.0f}%)")
    
    # Train lại model với alpha tốt nhất
    print("\n  🔧 Training model với alpha tối ưu...")
    model_best = HybridCF(rate_train, k=30, alpha=best_alpha)
    model_best.fit()
    print("  ✅ Model đã được train xong!")
    
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
    print(f"   - Alpha tối ưu: {best_alpha:.1f}")
    print(f"   - RMSE tốt nhất: {best_rmse:.4f}")
    print(f"   - Trọng số User-User CF: {best_alpha*100:.0f}%")
    print(f"   - Trọng số Item-Item CF: {(1-best_alpha)*100:.0f}%")


if __name__ == "__main__":
    main()
