"""
Hybrid Recommendation System for Travel Activities
Kết hợp Collaborative Filtering (CF) và Content-Based Filtering (CB)

Adapted from rcm/final/hybrid_cf_cb.py
- Sửa để phù hợp với 20 categories từ backend NestJS
- Đọc dữ liệu từ CSV thay vì ml-100k dataset
- Xuất kết quả recommendation ra CSV
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
import os
import pickle
import argparse
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
# CONTENT-BASED FILTERING
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
        ids = np.where(y == user_id)[0]
        item_ids = rate_matrix[ids, 1]
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
    
    def recommend(self, u, n_items, top_n=10):
        """Recommend top N items cho user u"""
        # Tìm các items mà user chưa rate
        ids = np.where(self.rate_train[:, 0] == u)[0]
        items_rated_by_u = self.rate_train[ids, 1].tolist()
        
        predictions = []
        
        for i in range(n_items):
            if i not in items_rated_by_u:
                try:
                    rating = self.pred(u, i)
                    predictions.append((i, rating))
                except:
                    continue
        
        # Sort theo rating giảm dần
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_n]


# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

def load_ratings_from_csv(file_path):
    """
    Load ratings từ CSV file (format: user_id, activity_id, rating, timestamp)
    """
    print(f"📂 Đang load ratings từ {file_path}...")
    
    try:
        # Đọc CSV (tab-separated hoặc comma-separated)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, sep=',', header=0)
        else:
            # Format ml-100k (tab-separated, no header)
            df = pd.read_csv(file_path, sep='\t', header=None, 
                           names=['user_id', 'activity_id', 'rating', 'timestamp'])
        
        # Chuyển đổi sang numpy array (user_id, activity_id, rating)
        ratings = df[['user_id', 'activity_id', 'rating']].to_numpy()
        
        # Đảm bảo IDs bắt đầu từ 0
        ratings[:, 0] = ratings[:, 0] - 1  # user_id: 1-based -> 0-based
        ratings[:, 1] = ratings[:, 1] - 1  # activity_id: 1-based -> 0-based
        
        print(f"✅ Đã load {ratings.shape[0]} ratings")
        return ratings
        
    except Exception as e:
        print(f"❌ Lỗi khi load ratings: {e}")
        raise


def load_activities_from_csv(file_path):
    """
    Load activities từ CSV file
    Format từ backend: activity_id, category_id, cat0, cat1, ..., cat19
    (Tổng cộng 22 cột: activity_id, category_id, và 20 binary category columns)
    """
    print(f"📂 Đang load activities từ {file_path}...")
    
    try:
        df = pd.read_csv(file_path, sep=',', header=0)
        
        # Lấy activity_id (cột đầu tiên)
        activity_ids = df.iloc[:, 0].values
        
        # Format từ backend: activity_id, category_id, cat0, cat1, ..., cat19
        # Bỏ 2 cột đầu (activity_id, category_id), lấy 20 cột binary tiếp theo
        if df.shape[1] >= 22:
            # Lấy 20 cột binary từ cột index 2-21 (cat0 đến cat19)
            category_features = df.iloc[:, 2:22].values.astype(float)
        elif df.shape[1] == 2 and 'category_id' in df.columns:
            # Nếu chỉ có activity_id và category_id, chuyển sang one-hot encoding
            category_id = df['category_id'].values
            n_categories = 20
            category_features = np.zeros((len(category_id), n_categories))
            for i, cat_id in enumerate(category_id):
                cat_idx = int(cat_id) if not np.isnan(cat_id) else 0
                if 0 <= cat_idx < n_categories:
                    category_features[i, cat_idx] = 1
        else:
            raise ValueError(f"Format CSV không hợp lệ. Expected 22 columns, got {df.shape[1]}")
        
        print(f"✅ Đã load {len(activity_ids)} activities với {category_features.shape[1]} category features")
        return activity_ids, category_features
        
    except Exception as e:
        print(f"❌ Lỗi khi load activities: {e}")
        raise


def create_tfidf_features(category_features):
    """
    Tạo TF-IDF features từ category binary matrix
    Vì là binary features nên TF-IDF sẽ không thay đổi nhiều, nhưng vẫn áp dụng để chuẩn hóa
    """
    print("🔧 Đang tạo TF-IDF features...")
    
    # Áp dụng TF-IDF transformation
    transformer = TfidfTransformer(smooth_idf=True, norm='l2')
    tfidf = transformer.fit_transform(category_features).toarray()
    
    print(f"✅ TF-IDF shape: {tfidf.shape}")
    return tfidf


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hybrid Recommendation System')
    parser.add_argument('--retrain', action='store_true', 
                       help='Train lại model từ đầu (bỏ qua cache)')
    parser.add_argument('--ratings', type=str, default='ratings.csv',
                       help='Path to ratings CSV file (default: ratings.csv)')
    parser.add_argument('--activities', type=str, default='activities.csv',
                       help='Path to activities CSV file (default: activities.csv)')
    parser.add_argument('--output', type=str, default='recommendations.csv',
                       help='Path to output CSV file (default: recommendations.csv)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HYBRID RECOMMENDATION SYSTEM FOR TRAVEL ACTIVITIES")
    print("Kết hợp Collaborative Filtering + Content-Based Filtering")
    print("="*70)
    
    # ========================================================================
    # Load data từ CSV files
    # ========================================================================
    ratings_file = args.ratings
    activities_file = args.activities
    output_file = args.output
    force_retrain = args.retrain
    
    # Kiểm tra files có tồn tại không
    if not os.path.exists(ratings_file):
        print(f"\n❌ Không tìm thấy file {ratings_file}")
        print("   Vui lòng đảm bảo file ratings.csv có trong cùng thư mục")
        return
    
    if not os.path.exists(activities_file):
        print(f"\n❌ Không tìm thấy file {activities_file}")
        print("   Vui lòng đảm bảo file activities.csv có trong cùng thư mục")
        return
    
    # Load activities và category features TRƯỚC để biết activity IDs hợp lệ
    activity_ids, category_features = load_activities_from_csv(activities_file)
    
    # Load và filter ratings: chỉ giữ ratings cho activities còn tồn tại
    print(f"\n🔍 Đang lọc và map ratings để khớp với {len(activity_ids)} activities hợp lệ...")
    
    # Load lại ratings từ CSV để filter trước khi convert
    df_ratings_original = pd.read_csv(ratings_file, sep=',', header=0)
    valid_activity_ids_set = set([int(aid) for aid in activity_ids])
    
    # Filter ratings: chỉ giữ ratings có activity_id trong danh sách activities hợp lệ
    original_ratings_count = len(df_ratings_original)
    df_ratings_filtered = df_ratings_original[df_ratings_original['activity_id'].isin(valid_activity_ids_set)].copy()
    
    if len(df_ratings_filtered) < original_ratings_count:
        removed_count = original_ratings_count - len(df_ratings_filtered)
        print(f"   ⚠️  Đã loại bỏ {removed_count} ratings cho activities không tồn tại")
    
    if len(df_ratings_filtered) == 0:
        print(f"\n❌ Không còn ratings hợp lệ nào sau khi lọc!")
        print(f"   Kiểm tra lại file ratings.csv và activities.csv")
        return
    
    # Tạo mapping từ activity_id thực tế (từ CSV) sang index (0-based) trong activity_ids array
    activity_id_to_index_map = {int(activity_ids[i]): i for i in range(len(activity_ids))}
    
    # Map activity_id từ ID thực tế sang index (0-based)
    df_ratings_filtered['activity_id'] = df_ratings_filtered['activity_id'].map(activity_id_to_index_map)
    
    # Remove any rows where activity_id mapping failed (shouldn't happen after filter, but just in case)
    df_ratings_filtered = df_ratings_filtered.dropna(subset=['activity_id'])
    
    # Convert sang numpy array
    rate_train = df_ratings_filtered[['user_id', 'activity_id', 'rating']].to_numpy()
    
    # Convert user_id sang 0-based
    rate_train[:, 0] = rate_train[:, 0] - 1
    
    # Convert activity_id sang int (đã là index 0-based)
    rate_train[:, 1] = rate_train[:, 1].astype(int)
    
    print(f"   ✅ Còn lại {len(rate_train)} ratings hợp lệ sau khi lọc và map")
    
    # Tạo TF-IDF features
    tfidf = create_tfidf_features(category_features)
    
    # ========================================================================
    # Load hoặc Train Hybrid Model
    # ========================================================================
    print("\n" + "="*70)
    print("🔧 LOADING/TRAINING HYBRID MODEL")
    print("="*70)
    
    # Tạo cache file name dựa trên config
    model_cache_file = 'hybrid_model_cache.pkl'
    
    # Kiểm tra cache và load nếu có (trừ khi force retrain)
    hybrid_model = None
    
    if not force_retrain and os.path.exists(model_cache_file):
        print(f"📦 Tìm thấy model đã train sẵn: {model_cache_file}")
        print("   Đang load model từ cache...")
        print("   (Để train lại model, sử dụng flag --retrain)")
        
        try:
            with open(model_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            hybrid_model = cache_data.get('model')
            
            if hybrid_model:
                cached_info = cache_data.get('info', {})
                print(f"   ✅ Đã load model từ cache thành công!")
                if cached_info:
                    print(f"      - Số users khi train: {cached_info.get('n_users', 'N/A')}")
                    print(f"      - Số activities khi train: {cached_info.get('n_items', 'N/A')}")
                
                # Update model với dữ liệu mới (ratings và activities mới)
                print(f"   🔄 Đang cập nhật model với dữ liệu mới...")
                hybrid_model.rate_train = rate_train
                hybrid_model.tfidf = tfidf
                # Update trong các sub-models nếu cần
                if hasattr(hybrid_model, 'cf_model'):
                    hybrid_model.cf_model.Y_data = rate_train
                if hasattr(hybrid_model, 'cb_model'):
                    hybrid_model.cb_model.rate_train = rate_train
                    hybrid_model.cb_model.tfidf = tfidf
                print(f"      ✅ Đã cập nhật với dữ liệu mới")
            else:
                print(f"   ⚠️  Cache file không hợp lệ, sẽ train lại...")
                hybrid_model = None
        except Exception as e:
            print(f"   ⚠️  Lỗi khi load cache: {e}")
            print(f"      Sẽ train lại model mới...")
            hybrid_model = None
    elif force_retrain:
        print("   🔄 Flag --retrain được bật, sẽ train lại model từ đầu...")
    else:
        print("   📭 Không tìm thấy model cache, sẽ train model mới...")
    
    # Train model nếu chưa có hoặc force retrain
    if hybrid_model is None:
        print("\n   🔧 Bắt đầu train model mới...")
        
        # Sử dụng trọng số đã tối ưu (có thể điều chỉnh)
        weight_cf = 0.5  # Có thể thử các giá trị khác: 0.25, 0.5, 0.75
        
        hybrid_model = HybridCFCB(
            rate_train, 
            tfidf,
            cf_k=30, 
            cf_alpha=0.5,
            cb_alpha=0.01, 
            cb_l1_ratio=0.5,
            weight_cf=weight_cf
        )
        
        hybrid_model.fit()
        
        # Lưu model vào cache
        print(f"\n💾 Đang lưu model vào cache: {model_cache_file}...")
        try:
            cache_data = {
                'model': hybrid_model,
                'info': {
                    'n_users': int(np.max(rate_train[:, 0])) + 1,
                    'n_items': len(activity_ids),
                }
            }
            with open(model_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"   ✅ Đã lưu model vào cache thành công!")
            print(f"      Lần chạy sau sẽ tự động dùng model này (không cần train lại)")
        except Exception as e:
            print(f"   ⚠️  Không thể lưu cache: {e}")
            print(f"      Lần chạy sau sẽ phải train lại.")
    
    # ========================================================================
    # Generate Recommendations cho tất cả users
    # ========================================================================
    print("\n" + "="*70)
    print("🎯 GENERATING RECOMMENDATIONS")
    print("="*70)
    
    n_users = int(np.max(rate_train[:, 0])) + 1
    n_items = len(activity_ids)
    
    print(f"\n📊 Tổng số users: {n_users}")
    print(f"📊 Tổng số activities: {n_items}")
    print(f"📊 Sẽ recommend top 10 activities cho mỗi user...")
    
    all_recommendations = []
    
    for user_id in range(n_users):
        recommendations = hybrid_model.recommend(user_id, n_items, top_n=10)
        
        for activity_idx, predicted_rating in recommendations:
            # Chuyển đổi lại index về activity_id thực tế
            actual_activity_id = activity_ids[int(activity_idx)]
            actual_user_id = user_id + 1  # Chuyển về 1-based
            
            all_recommendations.append({
                'user_id': int(actual_user_id),
                'activity_id': int(actual_activity_id),
                'predicted_rating': float(predicted_rating)
            })
        
        if (user_id + 1) % 50 == 0:
            print(f"  Processed {user_id + 1}/{n_users} users...")
    
    # ========================================================================
    # Lưu kết quả ra CSV
    # ========================================================================
    print(f"\n💾 Đang lưu recommendations vào {output_file}...")
    
    df_recommendations = pd.DataFrame(all_recommendations)
    df_recommendations = df_recommendations.sort_values(['user_id', 'predicted_rating'], 
                                                        ascending=[True, False])
    
    # Lưu ra CSV
    df_recommendations.to_csv(output_file, index=False, sep=',')
    
    print(f"✅ Đã lưu {len(all_recommendations)} recommendations vào {output_file}")
    print(f"   Format: user_id, activity_id, predicted_rating")
    
    # Hiển thị thống kê
    print("\n" + "="*70)
    print("📊 THỐNG KÊ")
    print("="*70)
    print(f"   Tổng số recommendations: {len(all_recommendations)}")
    print(f"   Số users: {n_users}")
    print(f"   Trung bình recommendations/user: {len(all_recommendations) / n_users:.1f}")
    print(f"   Rating dự đoán trung bình: {df_recommendations['predicted_rating'].mean():.2f}")
    print(f"   Rating dự đoán cao nhất: {df_recommendations['predicted_rating'].max():.2f}")
    print(f"   Rating dự đoán thấp nhất: {df_recommendations['predicted_rating'].min():.2f}")
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    print(f"\n💡 File kết quả: {output_file}")
    print(f"   Backend NestJS có thể đọc file này để import vào database")


if __name__ == "__main__":
    main()

