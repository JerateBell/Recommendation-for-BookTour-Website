# ============================================================
# HYBRID COLLABORATIVE FILTERING (FINAL – CHUẨN ĐỒ ÁN)
# User-User CF + Item-Item CF
# Dataset: MovieLens 100K
# Evaluation: RMSE – tìm alpha tối ưu
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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
    # Load MovieLens 100K
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    train = pd.read_csv('../data/ml-100k/ub.base', sep='\t', names=cols)
    test = pd.read_csv('../data/ml-100k/ub.test', sep='\t', names=cols)

    # Convert to numpy & index from 0
    rate_train = train[['user_id', 'item_id', 'rating']].to_numpy() - [1, 1, 0]
    rate_test = test[['user_id', 'item_id', 'rating']].to_numpy() - [1, 1, 0]

    print('=' * 60)
    print('HYBRID COLLABORATIVE FILTERING – FINAL EXPERIMENT')
    print('=' * 60)

    results = {}
    for alpha in np.arange(0, 1.1, 0.1):
        alpha = round(alpha, 1)
        model = HybridCF(rate_train, k=30, alpha=alpha)
        model.fit()
        score = rmse(model, rate_test)
        results[alpha] = score
        print(f'Alpha={alpha:.1f} | RMSE={score:.4f}')

    best_alpha = min(results, key=results.get)

    print('\n🏆 BEST RESULT')
    print(f'Alpha tối ưu: {best_alpha:.1f}')
    print(f'RMSE thấp nhất: {results[best_alpha]:.4f}')
