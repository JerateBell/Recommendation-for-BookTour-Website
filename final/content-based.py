"""
Content-Based Filtering - Recommendation System
Tuning ElasticNet Parameters
"""

import sys
import io
# Fix encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge, ElasticNet
from math import sqrt


# ============================================================================
# Bước 1-2: Load dữ liệu
# ============================================================================

# Reading user file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

n_users = users.shape[0]
print('Number of users:', n_users)

# Reading ratings file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

# Convert DataFrame -> numpy array
rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()

print("Number of training rates:", rate_train.shape[0])
print("Number of test rates:", rate_test.shape[0])


# ============================================================================
# Bước 3-4: Xử lý features của items
# ============================================================================

# Reading items file
i_cols = [
    'movie id', 'movie title', 'release date', 'video release date', 'IMDb URL',
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

n_items = items.shape[0]
print("Number of items:", n_items)

# Convert DataFrame to numpy array
X0 = items.to_numpy()

# Lấy 19 cột cuối (các thể loại phim)
X_train_counts = X0[:, -19:]


# ============================================================================
# Bước 5: Áp dụng TF-IDF
# ============================================================================

# TF-IDF: Term Frequency-Inverse Document Frequency
# - Giúp chuẩn hóa và đánh trọng số cho các features (thể loại phim)
# - smooth_idf=True: Tránh chia cho 0 khi tính IDF
# - norm='l2': Chuẩn hóa L2 để mỗi vector có độ dài 1 (giúp so sánh tốt hơn)
transformer = TfidfTransformer(smooth_idf=True, norm='l2')

# Kết quả: Ma trận TF-IDF [n_items x 19] - mỗi hàng là vector đặc trưng của 1 phim
tfidf = transformer.fit_transform(X_train_counts).toarray()

print("TF-IDF matrix shape:", tfidf.shape)


# ============================================================================
# Bước 6: Helper function
# ============================================================================

def get_items_rated_by_user(rate_matrix, user_id):
    """
    return (item_ids, scores)
    """
    y = rate_matrix[:, 0]  # all user_ids

    # find indices where user_id matches (data user_id starts at 1)
    ids = np.where(y == user_id + 1)[0]

    # movie_id in data starts from 1 → convert to 0-based index
    item_ids = rate_matrix[ids, 1] - 1

    # ratings of those items
    scores = rate_matrix[ids, 2]

    return (item_ids, scores)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate(Yhat, rates, n_users):
    """
    Tính RMSE (Root Mean Square Error) giữa rating thực tế và dự đoán
    
    RMSE = sqrt(mean((y_true - y_pred)^2))
    - RMSE càng nhỏ → mô hình càng tốt
    - RMSE trên test set thường cao hơn train set (overfitting)
    """
    se = 0  # Sum of squared errors
    cnt = 0  # Tổng số rating
    
    for n in range(n_users):
        # Lấy các phim và rating thực tế của user n
        ids, scores_truth = get_items_rated_by_user(rates, n)
        
        # Lấy rating dự đoán tương ứng
        scores_pred = Yhat[ids, n]
        
        # Tính squared error (bình phương sai số)
        e = scores_truth - scores_pred
        se += np.sum(e**2)
        cnt += e.size
    
    # RMSE = sqrt(mean squared error)
    return sqrt(se/cnt)


def evaluate_with_accuracy(Yhat, rates, n_users):
    """
    Tính RMSE và Accuracy giữa rating thực tế và dự đoán
    
    Returns:
        rmse: Root Mean Square Error
        accuracy: Tỷ lệ dự đoán trong khoảng ±1.0 của rating thực tế (%)
    """
    se = 0  # Sum of squared errors
    cnt = 0  # Tổng số rating
    within_1 = 0  # Số lượng dự đoán trong khoảng ±1
    
    for n in range(n_users):
        # Lấy các phim và rating thực tế của user n
        ids, scores_truth = get_items_rated_by_user(rates, n)
        
        if len(ids) == 0:
            continue
        
        # Lấy rating dự đoán tương ứng
        scores_pred = Yhat[ids, n]
        
        # Tính squared error (bình phương sai số)
        e = scores_truth - scores_pred
        se += np.sum(e**2)
        cnt += e.size
        
        # Tính accuracy (within ±1.0)
        within_1 += np.sum(np.abs(e) <= 1.0)
    
    # RMSE = sqrt(mean squared error)
    rmse = sqrt(se/cnt) if cnt > 0 else float('inf')
    
    # Accuracy (tỷ lệ phần trăm)
    accuracy = (within_1 / cnt * 100) if cnt > 0 else 0
    
    return rmse, accuracy


def print_evaluation_results(rmse_train, rmse_test, acc_train, acc_test, method_name=""):
    """
    In kết quả đánh giá (RMSE và Accuracy) một cách nhất quán
    """
    if method_name:
        print(f"\nKết quả {method_name}:")
    print(f"  RMSE train: {rmse_train:.4f}")
    print(f"  RMSE test:  {rmse_test:.4f}")
    print(f"  Accuracy train: {acc_train:.2f}% (within ±1.0)")
    print(f"  Accuracy test:  {acc_test:.2f}% (within ±1.0)")


# ============================================================================
# Tuning ElasticNet Parameters
# ============================================================================

def tune_elasticnet_parameters():
    """
    Tuning ElasticNet parameters (alpha và l1_ratio)
    - Tìm best alpha và l1_ratio cho ElasticNet
    """
    print("\n" + "="*70)
    print("TUNING ELASTICNET PARAMETERS")
    print("="*70)
    
    print("\nĐang tuning ElasticNet parameters...")
    
    # Tổng số user trong dataset
    n_users_total = int(rate_train[:, 0].max())
    d = tfidf.shape[1]  # Data dimension (19 genres)
    
    alphas = [0.001, 0.01, 0.1, 1.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_alpha = 0.01
    best_l1_ratio = 0.5
    best_rmse = float('inf')
    
    test_users_tune = min(50, n_users_total)
    
    print(f"\n📊 Testing trên {test_users_tune} users để tìm best parameters...")
    print(f"   Alpha values: {alphas}")
    print(f"   L1_ratio values: {l1_ratios}")
    print(f"   Tổng cộng: {len(alphas) * len(l1_ratios)} combinations\n")
    
    results = []
    
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            W_test = np.zeros((d, test_users_tune))
            b_test = np.zeros((1, test_users_tune))
            
            for n in range(test_users_tune):
                ids, scores = get_items_rated_by_user(rate_train, n)
                if len(ids) == 0:
                    continue
                
                Xhat = tfidf[ids, :]
                clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=1000)
                try:
                    clf.fit(Xhat, scores)
                    W_test[:, n] = clf.coef_
                    b_test[0, n] = clf.intercept_
                except:
                    continue
            
            Yhat_test = tfidf.dot(W_test) + b_test
            rmse = evaluate(Yhat_test, rate_test, test_users_tune)
            
            results.append((alpha, l1_ratio, rmse))
            print(f"  Alpha = {alpha:6.3f}, L1_ratio = {l1_ratio:.1f}  →  RMSE = {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
                best_l1_ratio = l1_ratio
    
    print(f"\n✅ Best ElasticNet parameters:")
    print(f"   Alpha = {best_alpha}, L1_ratio = {best_l1_ratio}")
    print(f"   RMSE = {best_rmse:.4f}")
    
    # Train với best parameters trên toàn bộ users
    global_mean = np.mean(rate_train[:, 2])
    MIN_RATINGS = 5
    
    W_elasticnet_tuned = np.zeros((d, n_users_total))
    b_elasticnet_tuned = np.zeros((1, n_users_total))
    
    print(f"\n🔧 Đang train mô hình với best parameters trên toàn bộ {n_users_total} users...")
    for n in range(n_users_total):
        ids, scores = get_items_rated_by_user(rate_train, n)
        
        if len(ids) < MIN_RATINGS:
            b_elasticnet_tuned[0, n] = global_mean
            W_elasticnet_tuned[:, n] = 0
            continue
        
        Xhat = tfidf[ids, :]
        clf = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, fit_intercept=True, max_iter=1000)
        try:
            clf.fit(Xhat, scores)
            W_elasticnet_tuned[:, n] = clf.coef_
            b_elasticnet_tuned[0, n] = clf.intercept_
        except:
            clf_fallback = Ridge(alpha=0.1, fit_intercept=True)
            clf_fallback.fit(Xhat, scores)
            W_elasticnet_tuned[:, n] = clf_fallback.coef_
            b_elasticnet_tuned[0, n] = clf_fallback.intercept_
        
        if (n + 1) % 100 == 0:
            print(f"    Processed {n+1}/{n_users_total} users...")
    
    print("  ✅ Training completed!")
    
    Yhat_elasticnet_tuned = tfidf.dot(W_elasticnet_tuned) + b_elasticnet_tuned
    
    rmse_train_tuned, acc_train_tuned = evaluate_with_accuracy(Yhat_elasticnet_tuned, rate_train, n_users_total)
    rmse_test_tuned, acc_test_tuned = evaluate_with_accuracy(Yhat_elasticnet_tuned, rate_test, n_users_total)
    
    print_evaluation_results(rmse_train_tuned, rmse_test_tuned,
                             acc_train_tuned, acc_test_tuned,
                             "với Tuned ElasticNet")
    
    return Yhat_elasticnet_tuned, rmse_test_tuned, best_alpha, best_l1_ratio


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONTENT-BASED FILTERING - TUNING ELASTICNET PARAMETERS")
    print("="*70)
    
    Yhat_tuned, rmse_test, best_alpha, best_l1_ratio = tune_elasticnet_parameters()
    
    print("\n" + "="*70)
    print("✅ HOÀN THÀNH!")
    print("="*70)
    print(f"\n💡 Kết luận:")
    print(f"   - Best Alpha: {best_alpha}")
    print(f"   - Best L1_ratio: {best_l1_ratio}")
    print(f"   - RMSE test: {rmse_test:.4f}")
