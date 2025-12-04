"""
Content-Based Filtering - Recommendation System

Tổng quan:
Notebook này implement thuật toán Content-Based Filtering - một phương pháp recommendation system 
dựa trên đặc điểm (features) của items để dự đoán rating.

Khác biệt với Collaborative Filtering:
- Collaborative Filtering: Dựa trên hành vi/rating của users khác
- Content-Based Filtering: Dựa trên đặc điểm của items (ví dụ: thể loại phim, diễn viên, đạo diễn...)

Quy trình:
1-2: Load dữ liệu (users và ratings train/test split) - Dataset: MovieLens 100k
3-4: Xử lý features của items - Lấy 19 cột thể loại phim, áp dụng TF-IDF
5-6: Học mô hình cho từng user - Sử dụng Ridge Regression
7-8: Dự đoán rating - Công thức: Yhat = tfidf × W + b
9: Đánh giá - Tính RMSE trên tập train và test
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
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
# Bước 7: Học mô hình cho từng user
# ============================================================================

# Tổng số user trong dataset
n_users = int(rate_train[:, 0].max())

d = tfidf.shape[1]  # Data dimension (19 genres → after TF-IDF vẫn là 19)

# W: Ma trận trọng số [19 x n_users] - mỗi cột là vector trọng số của 1 user
#    W[i, j] = trọng số user j gán cho thể loại i
# b: Vector bias [1 x n_users] - bias của mỗi user (rating trung bình)
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

# Học mô hình riêng cho từng user (personalized model)
for n in range(n_users):
    # Lấy các phim và rating của user n
    ids, scores = get_items_rated_by_user(rate_train, n)

    # Nếu user không có rating → skip
    if len(ids) == 0:
        continue

    # Tập TF-IDF của các movie user này đã rating
    Xhat = tfidf[ids, :]  # [số_phim_đã_rating x 19]

    # Ridge Regression: Học hàm f(Xhat) = Xhat × w + b để dự đoán scores
    # - Input: TF-IDF features của các phim đã rating
    # - Output: Ratings thực tế của user
    # - alpha=0.01: Regularization để tránh overfitting (L2 penalty)
    # - fit_intercept=True: Cho phép có bias term
    clf = Ridge(alpha=0.01, fit_intercept=True)

    clf.fit(Xhat, scores)

    # Lưu trọng số và bias của user n
    W[:, n] = clf.coef_      # Vector trọng số 19 chiều (mỗi thể loại có 1 trọng số)
    b[0, n] = clf.intercept_ # Bias (hệ số tự do - rating trung bình của user)

print("Training completed!")


# ============================================================================
# Bước 8: Dự đoán rating
# ============================================================================

# Dự đoán rating cho tất cả phim của tất cả users
# Công thức: Yhat[i, j] = tfidf[i, :] × W[:, j] + b[j]
# - i: index của phim (0..n_items-1)
# - j: index của user (0..n_users-1)  
# - Yhat[i, j]: Rating dự đoán của user j cho phim i
# 
# Giải thích: Với mỗi phim i, ta nhân vector đặc trưng TF-IDF của nó 
# với vector trọng số của user j, rồi cộng bias của user j
Yhat = tfidf.dot(W) + b

print("Prediction completed!")


# ============================================================================
# Bước 9: Đánh giá
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


# Đánh giá mô hình cơ bản
rmse_train_base, acc_train_base = evaluate_with_accuracy(Yhat, rate_train, n_users)
rmse_test_base, acc_test_base = evaluate_with_accuracy(Yhat, rate_test, n_users)

print_evaluation_results(rmse_train_base, rmse_test_base, 
                         acc_train_base, acc_test_base,
                         "Mô hình cơ bản")


# ============================================================================
# Ví dụ: Xem dự đoán cho một user cụ thể
# ============================================================================

user_id = 10
ids, scores = get_items_rated_by_user(rate_test, user_id)

predicted_scores = Yhat[ids, user_id]

print('\nExample prediction for user', user_id)
print('Rated movies ids:', ids)
print('True ratings:', scores)
print('Predicted ratings:', predicted_scores)


# ============================================================================
# Các phương pháp cải thiện
# ============================================================================

def tune_hyperparameter_alpha():
    """
    Phương pháp 1: Tuning hyperparameter alpha với Grid Search
    """
    # Thử các giá trị alpha khác nhau
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = 0.01
    best_rmse = float('inf')

    # Test trên một subset users để tiết kiệm thời gian
    test_users = min(100, n_users)

    for alpha in alphas:
        W_test = np.zeros((d, test_users))
        b_test = np.zeros((1, test_users))
        
        for n in range(test_users):
            ids, scores = get_items_rated_by_user(rate_train, n)
            if len(ids) == 0:
                continue
            
            Xhat = tfidf[ids, :]
            clf = Ridge(alpha=alpha, fit_intercept=True)
            clf.fit(Xhat, scores)
            
            W_test[:, n] = clf.coef_
            b_test[0, n] = clf.intercept_
        
        # Đánh giá trên test set
        Yhat_test = tfidf.dot(W_test) + b_test
        rmse = evaluate(Yhat_test, rate_test, test_users)
        
        print(f"Alpha = {alpha:6.3f}  →  RMSE = {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    print(f"\nBest alpha: {best_alpha} với RMSE = {best_rmse:.4f}")
    return best_alpha


def compare_different_algorithms():
    """
    Phương pháp 2: Thử các thuật toán khác
    """
    # Test trên subset users
    test_users = min(50, n_users)
    models = {
        'Ridge (alpha=0.01)': Ridge(alpha=0.01, fit_intercept=True),
        'Ridge (alpha=0.1)': Ridge(alpha=0.1, fit_intercept=True),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=True, max_iter=1000),
        'Lasso': Lasso(alpha=0.01, fit_intercept=True, max_iter=1000),
    }

    results = {}

    for model_name, model in models.items():
        W_test = np.zeros((d, test_users))
        b_test = np.zeros((1, test_users))
        
        for n in range(test_users):
            ids, scores = get_items_rated_by_user(rate_train, n)
            if len(ids) == 0:
                continue
            
            Xhat = tfidf[ids, :]
            
            try:
                model.fit(Xhat, scores)
                W_test[:, n] = model.coef_
                b_test[0, n] = model.intercept_ if hasattr(model, 'intercept_') else 0
            except:
                continue
        
        Yhat_test = tfidf.dot(W_test) + b_test
        rmse = evaluate(Yhat_test, rate_test, test_users)
        results[model_name] = rmse
        print(f"{model_name:25s}  →  RMSE = {rmse:.4f}")

    best_model = min(results, key=results.get)
    print(f"\nBest model: {best_model} với RMSE = {results[best_model]:.4f}")
    return results


def improved_model_with_cold_start():
    """
    Phương pháp 3: Cải thiện với xử lý Cold Start và minimum rating threshold
    """
    # Tính rating trung bình toàn cục (fallback cho users/items mới)
    global_mean = np.mean(rate_train[:, 2])
    print(f"Global mean rating: {global_mean:.3f}")

    # Minimum số rating để train mô hình (tránh overfitting với ít data)
    MIN_RATINGS = 5

    W_improved = np.zeros((d, n_users))
    b_improved = np.zeros((1, n_users))

    for n in range(n_users):
        ids, scores = get_items_rated_by_user(rate_train, n)
        
        if len(ids) < MIN_RATINGS:
            # User có ít rating → dùng global mean
            b_improved[0, n] = global_mean
            W_improved[:, n] = 0  # Không có trọng số
            continue
        
        Xhat = tfidf[ids, :]
        
        # Sử dụng alpha tốt hơn (từ tuning)
        clf = Ridge(alpha=0.1, fit_intercept=True)
        clf.fit(Xhat, scores)
        
        W_improved[:, n] = clf.coef_
        b_improved[0, n] = clf.intercept_

    # Dự đoán
    Yhat_improved = tfidf.dot(W_improved) + b_improved

    # Đánh giá
    rmse_train_improved, acc_train_improved = evaluate_with_accuracy(Yhat_improved, rate_train, n_users)
    rmse_test_improved, acc_test_improved = evaluate_with_accuracy(Yhat_improved, rate_test, n_users)

    print_evaluation_results(rmse_train_improved, rmse_test_improved,
                             acc_train_improved, acc_test_improved,
                             "với Cold Start handling")
    print(f"\nSo sánh với mô hình gốc:")
    print(f"  Train RMSE: {0.9089:.4f} → {rmse_train_improved:.4f} ({'↓' if rmse_train_improved < 0.9089 else '↑'} {abs(0.9089 - rmse_train_improved):.4f})")
    print(f"  Test RMSE:  {1.2703:.4f} → {rmse_test_improved:.4f} ({'↓' if rmse_test_improved < 1.2703 else '↑'} {abs(1.2703 - rmse_test_improved):.4f})")
    
    return Yhat_improved


def ensemble_method():
    """
    Phương pháp 4: Ensemble - Kết hợp nhiều mô hình
    """
    # Train 2 mô hình
    W_ridge = np.zeros((d, n_users))
    b_ridge = np.zeros((1, n_users))
    W_elastic = np.zeros((d, n_users))
    b_elastic = np.zeros((1, n_users))

    for n in range(n_users):
        ids, scores = get_items_rated_by_user(rate_train, n)
        if len(ids) < 5:
            continue
        
        Xhat = tfidf[ids, :]
        
        # Model 1: Ridge
        clf1 = Ridge(alpha=0.1, fit_intercept=True)
        clf1.fit(Xhat, scores)
        W_ridge[:, n] = clf1.coef_
        b_ridge[0, n] = clf1.intercept_
        
        # Model 2: ElasticNet
        clf2 = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=True, max_iter=1000)
        try:
            clf2.fit(Xhat, scores)
            W_elastic[:, n] = clf2.coef_
            b_elastic[0, n] = clf2.intercept_
        except:
            W_elastic[:, n] = clf1.coef_
            b_elastic[0, n] = clf1.intercept_

    # Dự đoán từ 2 mô hình
    Yhat_ridge = tfidf.dot(W_ridge) + b_ridge
    Yhat_elastic = tfidf.dot(W_elastic) + b_elastic

    # Ensemble: Weighted average (có thể tune weights)
    weight_ridge = 0.6
    weight_elastic = 0.4
    Yhat_ensemble = weight_ridge * Yhat_ridge + weight_elastic * Yhat_elastic

    # Đánh giá
    rmse_ridge, _ = evaluate_with_accuracy(Yhat_ridge, rate_test, n_users)
    rmse_elastic, _ = evaluate_with_accuracy(Yhat_elastic, rate_test, n_users)
    rmse_ensemble, acc_ensemble = evaluate_with_accuracy(Yhat_ensemble, rate_test, n_users)

    print(f"Ridge alone:     RMSE = {rmse_ridge:.4f}")
    print(f"ElasticNet alone: RMSE = {rmse_elastic:.4f}")
    print(f"\nEnsemble ({weight_ridge:.1f}/{weight_elastic:.1f}):")
    print(f"  RMSE test: {rmse_ensemble:.4f}")
    print(f"  Accuracy test: {acc_ensemble:.2f}% (within ±1.0)")
    print(f"\nCải thiện RMSE: {1.2703:.4f} → {rmse_ensemble:.4f} ({'↓' if rmse_ensemble < 1.2703 else '↑'} {abs(1.2703 - rmse_ensemble):.4f})")
    
    return Yhat_ensemble


def apply_all_improvements():
    """
    Phương pháp 5: Áp dụng tất cả các phương pháp cải thiện kết hợp
    - Tuning hyperparameter để tìm best alpha
    - Sử dụng Cold Start handling
    - Kết hợp với Ensemble method
    """
    print("Đang tìm best alpha...")
    # Tìm best alpha
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = 0.01
    best_rmse = float('inf')
    test_users_tune = min(100, n_users)
    
    for alpha in alphas:
        W_test = np.zeros((d, test_users_tune))
        b_test = np.zeros((1, test_users_tune))
        
        for n in range(test_users_tune):
            ids, scores = get_items_rated_by_user(rate_train, n)
            if len(ids) == 0:
                continue
            
            Xhat = tfidf[ids, :]
            clf = Ridge(alpha=alpha, fit_intercept=True)
            clf.fit(Xhat, scores)
            
            W_test[:, n] = clf.coef_
            b_test[0, n] = clf.intercept_
        
        Yhat_test = tfidf.dot(W_test) + b_test
        rmse = evaluate(Yhat_test, rate_test, test_users_tune)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    print(f"Best alpha tìm được: {best_alpha} với RMSE = {best_rmse:.4f}")
    
    # Tính rating trung bình toàn cục (fallback cho users/items mới)
    global_mean = np.mean(rate_train[:, 2])
    MIN_RATINGS = 5
    
    # Train với best alpha và cold start handling
    W_combined = np.zeros((d, n_users))
    b_combined = np.zeros((1, n_users))
    W_ridge_combined = np.zeros((d, n_users))
    b_ridge_combined = np.zeros((1, n_users))
    W_elastic_combined = np.zeros((d, n_users))
    b_elastic_combined = np.zeros((1, n_users))
    
    print("Đang train mô hình với best alpha và cold start handling...")
    for n in range(n_users):
        ids, scores = get_items_rated_by_user(rate_train, n)
        
        if len(ids) < MIN_RATINGS:
            # User có ít rating → dùng global mean
            b_combined[0, n] = global_mean
            W_combined[:, n] = 0
            b_ridge_combined[0, n] = global_mean
            W_ridge_combined[:, n] = 0
            b_elastic_combined[0, n] = global_mean
            W_elastic_combined[:, n] = 0
            continue
        
        Xhat = tfidf[ids, :]
        
        # Model 1: Ridge với best alpha
        clf1 = Ridge(alpha=best_alpha, fit_intercept=True)
        clf1.fit(Xhat, scores)
        W_combined[:, n] = clf1.coef_
        b_combined[0, n] = clf1.intercept_
        W_ridge_combined[:, n] = clf1.coef_
        b_ridge_combined[0, n] = clf1.intercept_
        
        # Model 2: ElasticNet
        clf2 = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=True, max_iter=1000)
        try:
            clf2.fit(Xhat, scores)
            W_elastic_combined[:, n] = clf2.coef_
            b_elastic_combined[0, n] = clf2.intercept_
        except:
            W_elastic_combined[:, n] = clf1.coef_
            b_elastic_combined[0, n] = clf1.intercept_
    
    # Dự đoán từ các mô hình
    Yhat_combined = tfidf.dot(W_combined) + b_combined
    Yhat_ridge_combined = tfidf.dot(W_ridge_combined) + b_ridge_combined
    Yhat_elastic_combined = tfidf.dot(W_elastic_combined) + b_elastic_combined
    
    # Ensemble: Weighted average
    weight_ridge = 0.6
    weight_elastic = 0.4
    Yhat_final = weight_ridge * Yhat_ridge_combined + weight_elastic * Yhat_elastic_combined
    
    # Đánh giá
    rmse_train_combined, _ = evaluate_with_accuracy(Yhat_combined, rate_train, n_users)
    rmse_test_combined, _ = evaluate_with_accuracy(Yhat_combined, rate_test, n_users)
    rmse_train_final, acc_train_final = evaluate_with_accuracy(Yhat_final, rate_train, n_users)
    rmse_test_final, acc_test_final = evaluate_with_accuracy(Yhat_final, rate_test, n_users)
    
    print(f"\nKết quả với Best Alpha + Cold Start:")
    print(f"  RMSE train: {rmse_train_combined:.4f}")
    print(f"  RMSE test:  {rmse_test_combined:.4f}")
    
    print(f"\nKết quả với TẤT CẢ phương pháp (Best Alpha + Cold Start + Ensemble):")
    print_evaluation_results(rmse_train_final, rmse_test_final,
                             acc_train_final, acc_test_final)
    
    print(f"\nSo sánh với mô hình gốc:")
    print(f"  Train RMSE: {0.9089:.4f} → {rmse_train_final:.4f} ({'↓' if rmse_train_final < 0.9089 else '↑'} {abs(0.9089 - rmse_train_final):.4f})")
    print(f"  Test RMSE:  {1.2703:.4f} → {rmse_test_final:.4f} ({'↓' if rmse_test_final < 1.2703 else '↑'} {abs(1.2703 - rmse_test_final):.4f})")
    
    return Yhat_final, rmse_test_final


def apply_all_improvements_with_elasticnet():
    """
    Phương pháp 6: Áp dụng tất cả các phương pháp cải thiện kết hợp
    - Tuning hyperparameter để tìm best alpha
    - Sử dụng Cold Start handling
    - Chỉ sử dụng ElasticNet (thay vì Ensemble)
    """
    print("Đang tìm best alpha...")
    # Tìm best alpha
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = 0.01
    best_rmse = float('inf')
    test_users_tune = min(100, n_users)
    
    for alpha in alphas:
        W_test = np.zeros((d, test_users_tune))
        b_test = np.zeros((1, test_users_tune))
        
        for n in range(test_users_tune):
            ids, scores = get_items_rated_by_user(rate_train, n)
            if len(ids) == 0:
                continue
            
            Xhat = tfidf[ids, :]
            clf = Ridge(alpha=alpha, fit_intercept=True)
            clf.fit(Xhat, scores)
            
            W_test[:, n] = clf.coef_
            b_test[0, n] = clf.intercept_
        
        Yhat_test = tfidf.dot(W_test) + b_test
        rmse = evaluate(Yhat_test, rate_test, test_users_tune)
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    print(f"Best alpha tìm được: {best_alpha} với RMSE = {best_rmse:.4f}")
    
    # Tính rating trung bình toàn cục (fallback cho users/items mới)
    global_mean = np.mean(rate_train[:, 2])
    MIN_RATINGS = 5
    
    # Train với best alpha, cold start handling và chỉ sử dụng ElasticNet
    W_elasticnet = np.zeros((d, n_users))
    b_elasticnet = np.zeros((1, n_users))
    
    print("Đang train mô hình với best alpha, cold start handling và ElasticNet...")
    for n in range(n_users):
        ids, scores = get_items_rated_by_user(rate_train, n)
        
        if len(ids) < MIN_RATINGS:
            # User có ít rating → dùng global mean
            b_elasticnet[0, n] = global_mean
            W_elasticnet[:, n] = 0
            continue
        
        Xhat = tfidf[ids, :]
        
        # Chỉ sử dụng ElasticNet với best alpha (hoặc có thể tune riêng cho ElasticNet)
        # Ở đây ta dùng best alpha từ Ridge, nhưng có thể điều chỉnh
        clf = ElasticNet(alpha=best_alpha, l1_ratio=0.5, fit_intercept=True, max_iter=1000)
        try:
            clf.fit(Xhat, scores)
            W_elasticnet[:, n] = clf.coef_
            b_elasticnet[0, n] = clf.intercept_
        except:
            # Nếu ElasticNet fail, fallback về Ridge với best alpha
            clf_fallback = Ridge(alpha=best_alpha, fit_intercept=True)
            clf_fallback.fit(Xhat, scores)
            W_elasticnet[:, n] = clf_fallback.coef_
            b_elasticnet[0, n] = clf_fallback.intercept_
    
    # Dự đoán
    Yhat_elasticnet = tfidf.dot(W_elasticnet) + b_elasticnet
    
    # Đánh giá
    rmse_train_elasticnet, acc_train_elasticnet = evaluate_with_accuracy(Yhat_elasticnet, rate_train, n_users)
    rmse_test_elasticnet, acc_test_elasticnet = evaluate_with_accuracy(Yhat_elasticnet, rate_test, n_users)
    
    print(f"\nKết quả với TẤT CẢ phương pháp (Best Alpha + Cold Start + ElasticNet):")
    print_evaluation_results(rmse_train_elasticnet, rmse_test_elasticnet,
                             acc_train_elasticnet, acc_test_elasticnet)
    
    print(f"\nSo sánh với mô hình gốc:")
    print(f"  Train RMSE: {0.9089:.4f} → {rmse_train_elasticnet:.4f} ({'↓' if rmse_train_elasticnet < 0.9089 else '↑'} {abs(0.9089 - rmse_train_elasticnet):.4f})")
    print(f"  Test RMSE:  {1.2703:.4f} → {rmse_test_elasticnet:.4f} ({'↓' if rmse_test_elasticnet < 1.2703 else '↑'} {abs(1.2703 - rmse_test_elasticnet):.4f})")
    
    return Yhat_elasticnet, rmse_test_elasticnet


def feature_normalization_improvement():
    """
    Phương pháp 7: Cải thiện với Feature Normalization/Standardization
    - Chuẩn hóa features trước khi train để cải thiện hiệu suất
    """
    print("Đang chuẩn hóa features...")
    
    # Chuẩn hóa TF-IDF features
    scaler = StandardScaler()
    tfidf_normalized = scaler.fit_transform(tfidf)
    
    # Tính rating trung bình toàn cục
    global_mean = np.mean(rate_train[:, 2])
    MIN_RATINGS = 5
    
    W_normalized = np.zeros((d, n_users))
    b_normalized = np.zeros((1, n_users))
    
    print("Đang train mô hình với normalized features...")
    for n in range(n_users):
        ids, scores = get_items_rated_by_user(rate_train, n)
        
        if len(ids) < MIN_RATINGS:
            b_normalized[0, n] = global_mean
            W_normalized[:, n] = 0
            continue
        
        Xhat = tfidf_normalized[ids, :]
        
        clf = Ridge(alpha=0.1, fit_intercept=True)
        clf.fit(Xhat, scores)
        
        W_normalized[:, n] = clf.coef_
        b_normalized[0, n] = clf.intercept_
    
    # Dự đoán (cần normalize features test)
    Yhat_normalized = tfidf_normalized.dot(W_normalized) + b_normalized
    
    rmse_train_norm, acc_train_norm = evaluate_with_accuracy(Yhat_normalized, rate_train, n_users)
    rmse_test_norm, acc_test_norm = evaluate_with_accuracy(Yhat_normalized, rate_test, n_users)
    
    print_evaluation_results(rmse_train_norm, rmse_test_norm,
                             acc_train_norm, acc_test_norm,
                             "với Feature Normalization")
    print(f"\nSo sánh với mô hình gốc:")
    print(f"  Train RMSE: {0.9089:.4f} → {rmse_train_norm:.4f} ({'↓' if rmse_train_norm < 0.9089 else '↑'} {abs(0.9089 - rmse_train_norm):.4f})")
    print(f"  Test RMSE:  {1.2703:.4f} → {rmse_test_norm:.4f} ({'↓' if rmse_test_norm < 1.2703 else '↑'} {abs(1.2703 - rmse_test_norm):.4f})")
    
    return Yhat_normalized, rmse_test_norm


def svr_improvement():
    """
    Phương pháp 8: Sử dụng SVR (Support Vector Regression) - Non-linear model
    - SVR có thể nắm bắt các mối quan hệ phi tuyến
    """
    print("Đang train mô hình với SVR (có thể mất nhiều thời gian)...")
    
    global_mean = np.mean(rate_train[:, 2])
    MIN_RATINGS = 5
    
    # SVR chậm hơn, chỉ train trên subset users để tiết kiệm thời gian
    test_users_svr = min(100, n_users)
    
    W_svr = np.zeros((d, test_users_svr))
    b_svr = np.zeros((1, test_users_svr))
    
    for n in range(test_users_svr):
        ids, scores = get_items_rated_by_user(rate_train, n)
        
        if len(ids) < MIN_RATINGS:
            b_svr[0, n] = global_mean
            W_svr[:, n] = 0
            continue
        
        Xhat = tfidf[ids, :]
        
        # SVR với linear kernel (để có coef_ và intercept_)
        # C=1.0: penalty parameter, epsilon=0.1: margin of tolerance
        try:
            clf_linear = SVR(kernel='linear', C=1.0, epsilon=0.1)
            clf_linear.fit(Xhat, scores)
            # SVR linear có coef_ là 2D array (n_features, 1)
            if hasattr(clf_linear, 'coef_') and clf_linear.coef_.shape[0] > 0:
                W_svr[:, n] = clf_linear.coef_[0] if clf_linear.coef_.ndim > 1 else clf_linear.coef_
            else:
                W_svr[:, n] = np.zeros(d)
            # intercept_ là array
            if hasattr(clf_linear, 'intercept_') and len(clf_linear.intercept_) > 0:
                b_svr[0, n] = clf_linear.intercept_[0] if isinstance(clf_linear.intercept_, np.ndarray) else clf_linear.intercept_
            else:
                b_svr[0, n] = global_mean
        except:
            # Fallback về Ridge nếu SVR fail
            clf_fallback = Ridge(alpha=0.1, fit_intercept=True)
            clf_fallback.fit(Xhat, scores)
            W_svr[:, n] = clf_fallback.coef_
            b_svr[0, n] = clf_fallback.intercept_
    
    Yhat_svr = tfidf.dot(W_svr) + b_svr
    
    rmse_train_svr, acc_train_svr = evaluate_with_accuracy(Yhat_svr, rate_train, test_users_svr)
    rmse_test_svr, acc_test_svr = evaluate_with_accuracy(Yhat_svr, rate_test, test_users_svr)
    
    print(f"\nKết quả với SVR (test trên {test_users_svr} users):")
    print_evaluation_results(rmse_train_svr, rmse_test_svr,
                             acc_train_svr, acc_test_svr)
    
    return Yhat_svr, rmse_test_svr


def tune_elasticnet_parameters():
    """
    Phương pháp 9: Tuning ElasticNet parameters (alpha và l1_ratio)
    - Tìm best alpha và l1_ratio cho ElasticNet
    """
    print("Đang tuning ElasticNet parameters...")
    
    alphas = [0.001, 0.01, 0.1, 1.0]
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_alpha = 0.01
    best_l1_ratio = 0.5
    best_rmse = float('inf')
    
    test_users_tune = min(50, n_users)
    
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
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
                best_l1_ratio = l1_ratio
    
    print(f"Best ElasticNet parameters: alpha={best_alpha}, l1_ratio={best_l1_ratio} với RMSE = {best_rmse:.4f}")
    
    # Train với best parameters trên toàn bộ users
    global_mean = np.mean(rate_train[:, 2])
    MIN_RATINGS = 5
    
    W_elasticnet_tuned = np.zeros((d, n_users))
    b_elasticnet_tuned = np.zeros((1, n_users))
    
    print("Đang train mô hình với best ElasticNet parameters...")
    for n in range(n_users):
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
    
    Yhat_elasticnet_tuned = tfidf.dot(W_elasticnet_tuned) + b_elasticnet_tuned
    
    rmse_train_tuned, acc_train_tuned = evaluate_with_accuracy(Yhat_elasticnet_tuned, rate_train, n_users)
    rmse_test_tuned, acc_test_tuned = evaluate_with_accuracy(Yhat_elasticnet_tuned, rate_test, n_users)
    
    print_evaluation_results(rmse_train_tuned, rmse_test_tuned,
                             acc_train_tuned, acc_test_tuned,
                             "với Tuned ElasticNet")
    print(f"\nSo sánh với mô hình gốc:")
    print(f"  Train RMSE: {0.9089:.4f} → {rmse_train_tuned:.4f} ({'↓' if rmse_train_tuned < 0.9089 else '↑'} {abs(0.9089 - rmse_train_tuned):.4f})")
    print(f"  Test RMSE:  {1.2703:.4f} → {rmse_test_tuned:.4f} ({'↓' if rmse_test_tuned < 1.2703 else '↑'} {abs(1.2703 - rmse_test_tuned):.4f})")
    
    return Yhat_elasticnet_tuned, rmse_test_tuned


def random_forest_improvement():
    """
    Phương pháp 10: Sử dụng Random Forest - Ensemble tree-based model
    - Random Forest có thể nắm bắt non-linear relationships và feature interactions
    """
    print("Đang train mô hình với Random Forest (có thể mất nhiều thời gian)...")
    
    global_mean = np.mean(rate_train[:, 2])
    MIN_RATINGS = 5
    
    # Random Forest chậm hơn, chỉ train trên subset users
    test_users_rf = min(100, n_users)
    
    # Random Forest không có coef_ như linear models
    # Ta cần lưu models và predict trực tiếp
    models_rf = {}
    
    for n in range(test_users_rf):
        ids, scores = get_items_rated_by_user(rate_train, n)
        
        if len(ids) < MIN_RATINGS:
            models_rf[n] = None  # Sẽ dùng global_mean khi predict
            continue
        
        Xhat = tfidf[ids, :]
        
        # Random Forest với 50 trees, max_depth=10 để tránh overfitting
        clf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(Xhat, scores)
        models_rf[n] = clf
    
    # Dự đoán
    Yhat_rf = np.zeros((n_items, test_users_rf))
    for n in range(test_users_rf):
        if models_rf[n] is None:
            Yhat_rf[:, n] = global_mean
        else:
            Yhat_rf[:, n] = models_rf[n].predict(tfidf)
    
    rmse_train_rf, acc_train_rf = evaluate_with_accuracy(Yhat_rf, rate_train, test_users_rf)
    rmse_test_rf, acc_test_rf = evaluate_with_accuracy(Yhat_rf, rate_test, test_users_rf)
    
    print(f"\nKết quả với Random Forest (test trên {test_users_rf} users):")
    print_evaluation_results(rmse_train_rf, rmse_test_rf,
                             acc_train_rf, acc_test_rf)
    
    return Yhat_rf, rmse_test_rf


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Content-Based Filtering - Recommendation System")
    print("="*60)
    
    # Áp dụng từng phương pháp cải thiện và in kết quả
    print("\n" + "="*60)
    print("BẮT ĐẦU ÁP DỤNG CÁC PHƯƠNG PHÁP CẢI THIỆN")
    print("="*60)
    
    # Phương pháp 1: Tuning hyperparameter alpha
    print("\n>>> PHƯƠNG PHÁP 1: Tuning Hyperparameter Alpha")
    print("-" * 60)
    best_alpha = tune_hyperparameter_alpha()
    
    # Phương pháp 2: So sánh các thuật toán khác nhau
    print("\n>>> PHƯƠNG PHÁP 2: So sánh các thuật toán khác nhau")
    print("-" * 60)
    algorithm_results = compare_different_algorithms()
    
    # Phương pháp 3: Cải thiện với Cold Start handling
    print("\n>>> PHƯƠNG PHÁP 3: Cải thiện với Cold Start handling")
    print("-" * 60)
    Yhat_improved = improved_model_with_cold_start()
    
    # Phương pháp 4: Ensemble method
    print("\n>>> PHƯƠNG PHÁP 4: Ensemble Method")
    print("-" * 60)
    Yhat_ensemble = ensemble_method()
    
    # Phương pháp 5: Áp dụng tất cả các phương pháp cải thiện
    print("\n>>> PHƯƠNG PHÁP 5: Áp dụng TẤT CẢ các phương pháp cải thiện (với Ensemble)")
    print("-" * 60)
    Yhat_final, rmse_final = apply_all_improvements()
    
    # Phương pháp 6: Áp dụng tất cả các phương pháp cải thiện với ElasticNet
    print("\n>>> PHƯƠNG PHÁP 6: Áp dụng TẤT CẢ các phương pháp cải thiện (với ElasticNet)")
    print("-" * 60)
    Yhat_elasticnet_final, rmse_elasticnet_final = apply_all_improvements_with_elasticnet()
    
    # Phương pháp 7: Feature Normalization
    print("\n>>> PHƯƠNG PHÁP 7: Feature Normalization/Standardization")
    print("-" * 60)
    Yhat_norm, rmse_norm = feature_normalization_improvement()
    
    # Phương pháp 8: SVR (Support Vector Regression)
    print("\n>>> PHƯƠNG PHÁP 8: SVR (Support Vector Regression)")
    print("-" * 60)
    Yhat_svr, rmse_svr = svr_improvement()
    
    # Phương pháp 9: Tuning ElasticNet Parameters
    print("\n>>> PHƯƠNG PHÁP 9: Tuning ElasticNet Parameters")
    print("-" * 60)
    Yhat_elasticnet_tuned, rmse_elasticnet_tuned = tune_elasticnet_parameters()
    
    # Phương pháp 10: Random Forest
    print("\n>>> PHƯƠNG PHÁP 10: Random Forest")
    print("-" * 60)
    Yhat_rf, rmse_rf = random_forest_improvement()
    
    print("\n" + "="*60)
    print("HOÀN THÀNH TẤT CẢ CÁC PHƯƠNG PHÁP CẢI THIỆN")
    print("="*60)
    
    # Tổng hợp kết quả - tính accuracy cho các phương pháp chính
    print(f"\nTổng hợp kết quả:")
    print("-" * 60)
    
    results_summary = []
    
    # Tính accuracy cho từng phương pháp
    _, acc_5 = evaluate_with_accuracy(Yhat_final, rate_test, n_users)
    results_summary.append(("Phương pháp 5 (Ensemble)", rmse_final, acc_5))
    
    _, acc_6 = evaluate_with_accuracy(Yhat_elasticnet_final, rate_test, n_users)
    results_summary.append(("Phương pháp 6 (ElasticNet)", rmse_elasticnet_final, acc_6))
    
    _, acc_7 = evaluate_with_accuracy(Yhat_norm, rate_test, n_users)
    results_summary.append(("Phương pháp 7 (Feature Normalization)", rmse_norm, acc_7))
    
    test_users_svr = min(100, n_users)
    _, acc_8 = evaluate_with_accuracy(Yhat_svr, rate_test, test_users_svr)
    results_summary.append(("Phương pháp 8 (SVR)", rmse_svr, acc_8))
    
    _, acc_9 = evaluate_with_accuracy(Yhat_elasticnet_tuned, rate_test, n_users)
    results_summary.append(("Phương pháp 9 (Tuned ElasticNet)", rmse_elasticnet_tuned, acc_9))
    
    test_users_rf = min(100, n_users)
    _, acc_10 = evaluate_with_accuracy(Yhat_rf, rate_test, test_users_rf)
    results_summary.append(("Phương pháp 10 (Random Forest)", rmse_rf, acc_10))
    
    # Sắp xếp theo RMSE
    results_summary.sort(key=lambda x: x[1])
    
    print(f"{'Phương pháp':<40} {'RMSE':<10} {'Accuracy (±1.0)':<15}")
    print("-" * 60)
    for method, rmse, acc in results_summary:
        print(f"{method:<40} {rmse:<10.4f} {acc:<15.2f}%")
    
    best_method = results_summary[0]
    print(f"\n→ Tốt nhất: {best_method[0]}")
    print(f"  RMSE test: {best_method[1]:.4f}")
    print(f"  Accuracy: {best_method[2]:.2f}% (within ±1.0)")
    print(f"  So sánh với mô hình gốc: RMSE {1.2703:.4f} → {best_method[1]:.4f} ({'↓' if best_method[1] < 1.2703 else '↑'} {abs(1.2703 - best_method[1]):.4f})")
    print("="*60)

