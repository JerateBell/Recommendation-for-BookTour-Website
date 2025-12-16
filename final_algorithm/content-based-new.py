# ============================================================
# CONTENT-BASED FILTERING (REFactored – Chuẩn đồ án)
# Features: Category, Destination, Price, Duration, Description
# Model: ElasticNet / Ridge (fallback)
# Evaluation: RMSE, Accuracy (±1), Precision@K, Recall@K, NDCG@K
# ============================================================

import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import pickle
import os
from math import sqrt
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import train_test_split

# ============================================================
# 1. LOAD & SPLIT DATA
# ============================================================

def load_data(items_path, ratings_path):
    items = pd.read_csv(items_path)
    ratings = pd.read_csv(ratings_path)

    if 'activity_id' in ratings.columns:
        ratings = ratings.rename(columns={'activity_id': 'item_id'})

    # Split train/test theo user + thời gian
    train, test = [], []
    for uid in ratings['user_id'].unique():
        ur = ratings[ratings['user_id'] == uid].sort_values('timestamp')
        split = int(len(ur) * 0.8)
        train.append(ur.iloc[:split])
        test.append(ur.iloc[split:])

    train = pd.concat(train)
    test = pd.concat(test)

    # Map item_id → index
    item2idx = {iid: i for i, iid in enumerate(items['item_id'])}
    idx2item = {i: iid for iid, i in item2idx.items()}  # Reverse mapping
    for df in [train, test]:
        df['item_index'] = df['item_id'].map(item2idx)
        df.dropna(inplace=True)

    rate_train = train[['user_id', 'item_index', 'rating']].to_numpy()
    rate_test = test[['user_id', 'item_index', 'rating']].to_numpy()

    n_users = int(max(rate_train[:, 0].max(), rate_test[:, 0].max()))
    return items, rate_train, rate_test, n_users, item2idx, idx2item


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def build_item_features(items, encoder=None, scaler=None, tfidf=None):
    """Build item features. Nếu các preprocessor đã có thì dùng, không thì tạo mới"""
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat = encoder.fit_transform(items[['category', 'destination']])
    else:
        cat = encoder.transform(items[['category', 'destination']])
    
    if scaler is None:
        scaler = MinMaxScaler()
        num = scaler.fit_transform(items[['price', 'duration']])
    else:
        num = scaler.transform(items[['price', 'duration']])
    
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=300, stop_words='english')
        text = tfidf.fit_transform(items['description'].fillna('')).toarray()
    else:
        text = tfidf.transform(items['description'].fillna('')).toarray()

    X = np.hstack([cat, num, text])
    return X, encoder, scaler, tfidf


# ============================================================
# 3. UTILS
# ============================================================

def get_user_ratings(rate_matrix, user_id):
    mask = rate_matrix[:, 0] == user_id
    data = rate_matrix[mask]
    return data[:, 1].astype(int), data[:, 2]


# ============================================================
# 4. TRAIN CONTENT-BASED MODEL
# ============================================================

def train_cbf(X, rate_train, n_users, alpha=0.1, l1_ratio=0.5, min_ratings=3):
    Yhat = np.zeros((X.shape[0], n_users))
    global_mean = np.mean(rate_train[:, 2])
    models = {}

    for uid in range(1, n_users + 1):
        ids, ratings = get_user_ratings(rate_train, uid)

        if len(ids) < min_ratings:
            Yhat[:, uid - 1] = global_mean
            continue

        X_u = X[ids]
        try:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000)
            model.fit(X_u, ratings)
            preds = model.predict(X)

            if np.std(preds) < 0.01:
                raise ValueError

        except:
            model = Ridge(alpha=0.01)
            model.fit(X_u, ratings)
            preds = model.predict(X)

        models[uid] = model
        Yhat[:, uid - 1] = preds

    return models, Yhat


# ============================================================
# 5. EVALUATION
# ============================================================

def rmse_accuracy(Yhat, rates, n_users):
    se, cnt, within1 = 0, 0, 0
    for uid in range(1, n_users + 1):
        ids, y_true = get_user_ratings(rates, uid)
        if len(ids) == 0: continue
        y_pred = Yhat[ids, uid - 1]
        err = y_true - y_pred
        se += np.sum(err ** 2)
        within1 += np.sum(np.abs(err) <= 1)
        cnt += len(err)
    return sqrt(se / cnt), within1 / cnt * 100


def topk_metrics(Yhat, rates, n_users, k=10, threshold=4):
    P, R, N = [], [], []
    for uid in range(1, n_users + 1):
        ids, ratings = get_user_ratings(rates, uid)
        if len(ids) == 0: continue

        topk = np.argsort(Yhat[:, uid - 1])[::-1][:k]
        relevant = ids[ratings >= threshold]
        if len(relevant) == 0: continue

        hit = np.intersect1d(topk, relevant)
        P.append(len(hit) / k)
        R.append(len(hit) / len(relevant))

        rel = np.zeros(k)
        for i, idx in enumerate(topk):
            if idx in ids:
                rel[i] = ratings[ids.tolist().index(idx)]
        dcg = np.sum(rel / np.log2(np.arange(2, k + 2)))
        idcg = np.sum(np.sort(ratings)[::-1][:k] / np.log2(np.arange(2, min(k, len(ratings)) + 2)))
        N.append(dcg / idcg if idcg > 0 else 0)

    return np.mean(P), np.mean(R), np.mean(N)


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == '__main__':
    items, rate_train, rate_test, n_users, item2idx, idx2item = load_data(
        '../data/items.csv', '../data/ratings.csv'
    )

    X, encoder, scaler, tfidf = build_item_features(items)

    models, Yhat = train_cbf(X, rate_train, n_users)

    rmse_tr, acc_tr = rmse_accuracy(Yhat, rate_train, n_users)
    rmse_te, acc_te = rmse_accuracy(Yhat, rate_test, n_users)

    p, r, ndcg = topk_metrics(Yhat, rate_test, n_users)

    print('\n📊 CONTENT-BASED FILTERING RESULTS')
    print(f'RMSE train: {rmse_tr:.4f} | test: {rmse_te:.4f}')
    print(f'Accuracy train: {acc_tr:.2f}% | test: {acc_te:.2f}%')
    print(f'Precision@10: {p:.4f} | Recall@10: {r:.4f} | NDCG@10: {ndcg:.4f}')
    
    # Lưu model và các preprocessor
    model_dir = '../models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'models': models,  # Dictionary các model cho từng user
        'encoder': encoder,
        'scaler': scaler,
        'tfidf': tfidf,
        'item2idx': item2idx,
        'idx2item': idx2item,
        'X': X,  # Item features matrix
        'n_users': n_users,
        'n_items': len(items)
    }
    
    model_path = os.path.join(model_dir, 'content_based_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f'\n✅ Đã lưu model vào: {model_path}')
    print(f'   - Số lượng user models: {len(models)}')
    print(f'   - Số lượng items: {len(items)}')
    print(f'   - Item features shape: {X.shape}')
    print(f'   - Preprocessors: encoder, scaler, tfidf')
