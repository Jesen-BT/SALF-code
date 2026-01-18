import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores)
    exp_s = np.exp(scores)
    s = exp_s.sum()
    if s <= 0:
        return np.ones_like(scores) / len(scores)
    return exp_s / s


class _IIFBinary:


    def __init__(self, C=0.01, block_size=200, eps=1e-12):
        self.C = float(C)
        self.block_size = int(block_size)
        self.eps = float(eps)

        self.part_dims = []

        self.W_parts = []

        self.total_dim = 0


        self._buf_X = []
        self._buf_y = []

    def _ensure_parts(self, d: int):

        d = int(d)
        if self.total_dim == 0:

            self.part_dims = [d]
            self.W_parts = [np.zeros((d,), dtype=float)]
            self.total_dim = d
            return

        if d > self.total_dim:

            new_dim = d - self.total_dim
            self.part_dims.append(new_dim)
            self.W_parts.append(np.zeros((new_dim,), dtype=float))
            self.total_dim = d

    def _split_parts(self, x: np.ndarray):

        parts = []
        start = 0
        for pd in self.part_dims:
            end = start + pd
            parts.append(x[start:end])
            start = end
        return parts

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.total_dim == 0:
            return np.zeros((X.shape[0],), dtype=float)


        self._ensure_parts(X.shape[1])

        scores = np.zeros((X.shape[0],), dtype=float)
        for i in range(X.shape[0]):
            x = X[i]

            if x.shape[0] < self.total_dim:
                x = np.concatenate([x, np.zeros((self.total_dim - x.shape[0],), dtype=float)])
            parts = self._split_parts(x)
            s = 0.0
            for w, xp in zip(self.W_parts, parts):
                s += float(np.dot(w, xp))
            scores[i] = s
        return scores

    def _buffer_to_matrix(self, X_list, target_dim: int) -> np.ndarray:

        n = len(X_list)
        Xmat = np.zeros((n, target_dim), dtype=float)
        for i, xi in enumerate(X_list):
            xi = np.asarray(xi, dtype=float).ravel()
            d = min(xi.shape[0], target_dim)
            if d > 0:
                Xmat[i, :d] = xi[:d]
        return Xmat

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(1, -1)


        self._ensure_parts(X.shape[1])

        for xi, yi in zip(X, y):
            xi = np.asarray(xi, dtype=float).ravel()
            self._buf_X.append(xi)
            self._buf_y.append(int(yi))

            if len(self._buf_X) >= self.block_size:
                self._flush_update()

        return self

    def _flush_update(self):

        if len(self._buf_X) == 0:
            return

        Xb = self._buffer_to_matrix(self._buf_X, self.total_dim)
        yb = np.asarray(self._buf_y, dtype=int)


        self._ensure_parts(Xb.shape[1])


        l = np.zeros((len(self.part_dims),), dtype=float)
        g_parts = [np.zeros((pd,), dtype=float) for pd in self.part_dims]

        n = Xb.shape[0]
        for i in range(n):
            x = Xb[i]
            yi = float(yb[i])
            parts = self._split_parts(x)


            for j, (w, xp) in enumerate(zip(self.W_parts, parts)):
                margin = yi * float(np.dot(w, xp))
                loss_ij = max(0.0, 1.0 - margin)
                l[j] += loss_ij


                if loss_ij > 0.0:
                    g_parts[j] += yi * xp


        l = l / max(1, n)
        for j in range(len(g_parts)):
            g_parts[j] = g_parts[j] / max(1, n)


        denom = float(np.dot(l, l)) + self.eps
        taus = (self.C * l) / denom


        for j in range(len(self.W_parts)):
            self.W_parts[j] = self.W_parts[j] + taus[j] * g_parts[j]


        self._buf_X, self._buf_y = [], []


class IIFClassifier(BaseSKMObject, ClassifierMixin):


    def __init__(self, C=0.01, block_size=200):
        super().__init__()
        self.C = float(C)
        self.block_size = int(block_size)

        self.classes_ = None
        self.models_ = None
        self._is_initialized = False

    def partial_fit(self, X, y, classes=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        y = np.asarray(y, dtype=int).reshape(-1)

        if (not self._is_initialized) or (self.classes_ is None):
            if classes is None:

                self.classes_ = np.unique(y)
            else:
                self.classes_ = np.asarray(classes)
            self.models_ = {c: _IIFBinary(C=self.C, block_size=self.block_size) for c in self.classes_}
            self._is_initialized = True


        for c in self.classes_:
            y_bin = np.where(y == c, 1, -1)
            self.models_[c].partial_fit(X, y_bin)

        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        if proba.shape[1] == 0:
            return np.array([], dtype=int)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._is_initialized or self.classes_ is None:
            return np.zeros((X.shape[0], 0), dtype=float)


        scores = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for j, c in enumerate(self.classes_):
            scores[:, j] = self.models_[c].decision_function(X)


        P = np.zeros_like(scores)
        for i in range(scores.shape[0]):
            P[i] = _softmax(scores[i])
        return P








try:
    from skmultiflow.core import BaseSKMObject, ClassifierMixin
except Exception:

    class BaseSKMObject(object):
        pass

    class ClassifierMixin(object):
        pass


_EPS = 1e-12


def _softmax(z: np.ndarray) -> np.ndarray:

    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + _EPS)


@dataclass
class _OnlineMinMax:

    min_: float = np.inf
    max_: float = -np.inf

    def update(self, x: float):
        if np.isnan(x):
            return
        if x < self.min_:
            self.min_ = x
        if x > self.max_:
            self.max_ = x

    def bin_index(self, x: float, n_bins: int) -> int:

        if np.isnan(x):
            return 0
        if not np.isfinite(self.min_) or not np.isfinite(self.max_) or self.max_ <= self.min_ + _EPS:
            return 0

        x_clipped = min(max(x, self.min_), self.max_)

        ratio = (x_clipped - self.min_) / (self.max_ - self.min_ + _EPS)
        b = int(np.floor(ratio * n_bins))
        if b >= n_bins:
            b = n_bins - 1
        if b < 0:
            b = 0
        return b


class RAILClassifier(BaseSKMObject, ClassifierMixin):

    def __init__(
        self,
        n_bins: int = 10,
        lr: float = 0.01,
        n_gd_steps: int = 1,
        laplace: float = 1.0,
        redundancy_pair_sample: int = 0,
        random_state: int = 42,
    ):
        self.n_bins = int(n_bins)
        self.lr = float(lr)
        self.n_gd_steps = int(n_gd_steps)
        self.laplace = float(laplace)
        self.redundancy_pair_sample = int(redundancy_pair_sample)
        self.random_state = int(random_state)

        self._rng = np.random.RandomState(self.random_state)
        self.reset()


    def reset(self):
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0


        self.n_features_: int = 0


        self.class_count_: Optional[np.ndarray] = None

        self.feature_bin_count_: List[np.ndarray] = []


        self._xb_y_counts: List[np.ndarray] = []
        self._xb_counts: List[np.ndarray] = []
        self._y_counts: Optional[np.ndarray] = None


        self._xixj_counts: Dict[Tuple[int, int], np.ndarray] = {}
        self._xi_counts_for_pair: Dict[Tuple[int, int], np.ndarray] = {}
        self._xj_counts_for_pair: Dict[Tuple[int, int], np.ndarray] = {}
        self._pair_total: Dict[Tuple[int, int], float] = {}


        self._minmax: List[_OnlineMinMax] = []


        self.alpha_: np.ndarray = np.zeros(0, dtype=float)
        self.beta_: np.ndarray = np.zeros(0, dtype=float)
        self.gamma_: np.ndarray = np.zeros(0, dtype=float)


        self.n_seen_: int = 0
        return self

    def partial_fit(self, X, y, classes=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = y.reshape(1)
        if y.ndim > 1:
            y = y.ravel()

        if classes is not None and self.classes_ is None:
            self._init_classes(classes)


        if self.classes_ is None:
            self._init_classes(np.unique(y))


        if X.shape[1] > self.n_features_:
            self._expand_features(X.shape[1] - self.n_features_)

        for xi, yi in zip(X, y):
            self._partial_fit_one(xi, yi)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.classes_ is None or self.n_seen_ == 0:

            if self.classes_ is None:
                return np.ones((X.shape[0], 1), dtype=float)
            return np.ones((X.shape[0], self.n_classes_), dtype=float) / max(self.n_classes_, 1)


        if X.shape[1] > self.n_features_:
            self._expand_features(X.shape[1] - self.n_features_)

        proba = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for i, xi in enumerate(X):
            proba[i, :] = self._predict_proba_one(xi)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


    def _init_classes(self, classes):
        self.classes_ = np.array(classes)
        self.n_classes_ = len(self.classes_)
        self.class_count_ = np.zeros(self.n_classes_, dtype=float)
        self._y_counts = np.zeros(self.n_classes_, dtype=float)

    def _expand_features(self, add_d: int):

        if add_d <= 0:
            return
        old_d = self.n_features_
        new_d = old_d + add_d


        self.alpha_ = np.concatenate([self.alpha_, np.zeros(add_d, dtype=float)])
        self.beta_ = np.concatenate([self.beta_, np.zeros(add_d, dtype=float)])
        self.gamma_ = np.concatenate([self.gamma_, np.zeros(add_d, dtype=float)])


        for _ in range(add_d):
            self.feature_bin_count_.append(np.zeros((self.n_classes_, self.n_bins), dtype=float))
            self._xb_y_counts.append(np.zeros((self.n_bins, self.n_classes_), dtype=float))
            self._xb_counts.append(np.zeros(self.n_bins, dtype=float))
            self._minmax.append(_OnlineMinMax())

        self.n_features_ = new_d

    def _class_index(self, y_val) -> int:

        return int(np.where(self.classes_ == y_val)[0][0])

    def _discretize(self, x: np.ndarray) -> np.ndarray:

        bins = np.zeros(self.n_features_, dtype=int)
        for j in range(self.n_features_):
            v = x[j] if j < x.shape[0] else np.nan
            self._minmax[j].update(v)
            bins[j] = self._minmax[j].bin_index(v, self.n_bins)
        return bins

    def _partial_fit_one(self, x: np.ndarray, y_val):
        if x.ndim != 1:
            x = np.asarray(x).ravel()


        if x.shape[0] > self.n_features_:
            self._expand_features(x.shape[0] - self.n_features_)

        y_idx = self._class_index(y_val)
        xb = self._discretize(x)


        self.class_count_[y_idx] += 1.0
        self._y_counts[y_idx] += 1.0
        self.n_seen_ += 1

        for i in range(self.n_features_):
            b = int(xb[i])
            self.feature_bin_count_[i][y_idx, b] += 1.0
            self._xb_y_counts[i][b, y_idx] += 1.0
            self._xb_counts[i][b] += 1.0


        if self.n_features_ >= 2:
            if self.redundancy_pair_sample > 0:
                self._update_redundancy_sampled(xb, self.redundancy_pair_sample)
            else:
                self._update_redundancy_all_pairs(xb)


        for _ in range(self.n_gd_steps):
            self._update_weights_gd(x, y_idx, xb)

    def _update_redundancy_all_pairs(self, xb: np.ndarray):
        d = self.n_features_
        for i in range(d - 1):
            bi = int(xb[i])
            for j in range(i + 1, d):
                bj = int(xb[j])
                key = (i, j)
                if key not in self._xixj_counts:
                    self._xixj_counts[key] = np.zeros((self.n_bins, self.n_bins), dtype=float)
                    self._xi_counts_for_pair[key] = np.zeros(self.n_bins, dtype=float)
                    self._xj_counts_for_pair[key] = np.zeros(self.n_bins, dtype=float)
                    self._pair_total[key] = 0.0
                self._xixj_counts[key][bi, bj] += 1.0
                self._xi_counts_for_pair[key][bi] += 1.0
                self._xj_counts_for_pair[key][bj] += 1.0
                self._pair_total[key] += 1.0

    def _update_redundancy_sampled(self, xb: np.ndarray, n_pairs: int):
        d = self.n_features_

        if d < 2:
            return
        max_pairs = d * (d - 1) // 2
        n_pairs = int(min(max(n_pairs, 1), max_pairs))


        sampled = set()
        tries = 0
        while len(sampled) < n_pairs and tries < n_pairs * 20:
            i = int(self._rng.randint(0, d - 1))
            j = int(self._rng.randint(i + 1, d))
            sampled.add((i, j))
            tries += 1

        for (i, j) in sampled:
            bi = int(xb[i])
            bj = int(xb[j])
            key = (i, j)
            if key not in self._xixj_counts:
                self._xixj_counts[key] = np.zeros((self.n_bins, self.n_bins), dtype=float)
                self._xi_counts_for_pair[key] = np.zeros(self.n_bins, dtype=float)
                self._xj_counts_for_pair[key] = np.zeros(self.n_bins, dtype=float)
                self._pair_total[key] = 0.0
            self._xixj_counts[key][bi, bj] += 1.0
            self._xi_counts_for_pair[key][bi] += 1.0
            self._xj_counts_for_pair[key][bj] += 1.0
            self._pair_total[key] += 1.0

    def _mutual_information_x_y(self, i: int) -> float:

        xy = self._xb_y_counts[i]
        x = self._xb_counts[i]
        y = self._y_counts
        n = np.sum(y)
        if n <= 0:
            return 0.0


        pxy = xy / (n + _EPS)
        px = x / (n + _EPS)
        py = y / (n + _EPS)

        mi = 0.0
        for b in range(self.n_bins):
            for c in range(self.n_classes_):
                v = pxy[b, c]
                if v > 0:
                    mi += v * np.log((v + _EPS) / (px[b] * py[c] + _EPS))
        return float(mi)

    def _mutual_information_xi_xj(self, i: int, j: int) -> float:

        key = (i, j) if i < j else (j, i)
        if key not in self._xixj_counts:
            return 0.0

        n = self._pair_total.get(key, 0.0)
        if n <= 0:
            return 0.0

        xixj = self._xixj_counts[key]
        xi = self._xi_counts_for_pair[key]
        xj = self._xj_counts_for_pair[key]

        pxy = xixj / (n + _EPS)
        px = xi / (n + _EPS)
        py = xj / (n + _EPS)

        mi = 0.0
        for bi in range(self.n_bins):
            for bj in range(self.n_bins):
                v = pxy[bi, bj]
                if v > 0:
                    mi += v * np.log((v + _EPS) / (px[bi] * py[bj] + _EPS))
        return float(mi)

    def _compute_relevance_redundancy(self) -> Tuple[np.ndarray, np.ndarray]:

        d = self.n_features_
        rel = np.zeros(d, dtype=float)
        red = np.zeros(d, dtype=float)

        for i in range(d):
            rel[i] = self._mutual_information_x_y(i)


        for (i, j), _mat in self._xixj_counts.items():
            mij = self._mutual_information_xi_xj(i, j)
            red[i] += mij
            red[j] += mij

        return rel, red

    def _predict_proba_one(self, x: np.ndarray) -> np.ndarray:

        if x.ndim != 1:
            x = np.asarray(x).ravel()


        if x.shape[0] > self.n_features_:
            self._expand_features(x.shape[0] - self.n_features_)

        xb = self._discretize(x)


        total = np.sum(self.class_count_) + _EPS
        logp = np.log((self.class_count_ + self.laplace) / (total + self.laplace * self.n_classes_) + _EPS)


        for i in range(self.n_features_):
            b = int(xb[i])

            num = self.feature_bin_count_[i][:, b] + self.laplace
            den = self.class_count_ + self.laplace * self.n_bins
            loglik = np.log(num / (den + _EPS) + _EPS)
            logp = logp + self.gamma_[i] * loglik

        return _softmax(logp)

    def _update_weights_gd(self, x: np.ndarray, y_idx: int, xb: np.ndarray):

        rel, red = self._compute_relevance_redundancy()

        rel_n = rel / (np.max(rel) + _EPS) if np.max(rel) > 0 else rel
        red_n = red / (np.max(red) + _EPS) if np.max(red) > 0 else red


        total = np.sum(self.class_count_) + _EPS
        scores = np.log((self.class_count_ + self.laplace) / (total + self.laplace * self.n_classes_) + _EPS)


        loglik_mat = np.zeros((self.n_features_, self.n_classes_), dtype=float)
        for i in range(self.n_features_):
            b = int(xb[i])
            num = self.feature_bin_count_[i][:, b] + self.laplace
            den = self.class_count_ + self.laplace * self.n_bins
            loglik = np.log(num / (den + _EPS) + _EPS)
            loglik_mat[i, :] = loglik
            scores = scores + self.gamma_[i] * loglik

        p = _softmax(scores)
        t = np.zeros(self.n_classes_, dtype=float)
        t[y_idx] = 1.0


        diff = (p - t)

        grad_gamma = np.dot(loglik_mat, diff)


        self.gamma_ = self.gamma_ - self.lr * grad_gamma


        self.gamma_ = np.clip(self.gamma_, -5.0, 5.0)


        self.alpha_ = (1.0 - self.lr) * self.alpha_ + self.lr * rel_n
        self.beta_ = (1.0 - self.lr) * self.beta_ + self.lr * red_n


        pull = (self.alpha_ - self.beta_) - self.gamma_
        self.gamma_ = self.gamma_ + 0.1 * self.lr * pull


    def get_info(self):
        return (
            f"RAILClassifier(n_bins={self.n_bins}, lr={self.lr}, n_gd_steps={self.n_gd_steps}, "
            f"laplace={self.laplace}, redundancy_pair_sample={self.redundancy_pair_sample})"
        )




try:
    from skmultiflow.trees import HoeffdingTreeClassifier
except Exception:
    HoeffdingTreeClassifier = None


class IWEMclassifier:


    def __init__(
        self,
        base_estimator=None,
        chunk_size: int = 500,
        max_ensemble_size: int = 5,
        max_window_size: int = 500,
        random_state: int | None = None,
    ):
        if base_estimator is None:
            if HoeffdingTreeClassifier is None:
                raise ImportError(
                    "skmultiflow is not available, please install scikit-multiflow "
                    "or pass a custom incremental base_estimator."
                )
            base_estimator = HoeffdingTreeClassifier()

        self.base_estimator = base_estimator
        self.chunk_size = int(chunk_size)
        self.max_ensemble_size = int(max_ensemble_size)
        self.max_window_size = int(max_window_size)
        self.random_state = random_state


        self.models_: list = []
        self.cw_: np.ndarray | None = None
        self.eps_: np.ndarray | None = None


        self._buf_X: list[np.ndarray] = []
        self._buf_y: list[int] = []


        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None


        self._fitted_once = False


    def _clone_estimator(self):

        est = deepcopy(self.base_estimator)

        if self.random_state is not None and hasattr(est, "random_state"):
            try:
                est.random_state = self.random_state
            except Exception:
                pass
        return est

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _ensure_1d_y(self, y):
        y = np.asarray(y)
        if y.ndim == 0:
            y = y.reshape(1,)
        return y

    def _safe_predict_proba(self, model, X):

        X = self._ensure_2d(X)
        C = int(self.n_classes_)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            proba = np.asarray(proba)

            if proba.ndim == 1:

                if C == 2:
                    proba = np.vstack([1.0 - proba, proba]).T
                else:

                    preds = model.predict(X)
                    return self._hard_to_proba(preds, C)
            if proba.shape[1] != C:

                if hasattr(model, "classes_") and model.classes_ is not None:
                    aligned = np.zeros((proba.shape[0], C), dtype=float)
                    for j, cls in enumerate(model.classes_):
                        idx = int(np.where(self.classes_ == cls)[0][0])
                        aligned[:, idx] = proba[:, j]

                    s = aligned.sum(axis=1, keepdims=True)
                    s[s == 0] = 1.0
                    aligned /= s
                    return aligned
                preds = model.predict(X)
                return self._hard_to_proba(preds, C)

            s = proba.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return proba / s
        else:
            preds = model.predict(X)
            return self._hard_to_proba(preds, C)

    def _hard_to_proba(self, preds, C):
        preds = np.asarray(preds).reshape(-1)
        proba = np.zeros((preds.shape[0], C), dtype=float)
        for i, p in enumerate(preds):
            idx = int(np.where(self.classes_ == p)[0][0])
            proba[i, idx] = 1.0
        return proba

    def _update_weight_vector_one_instance(self, k: int, y_true_idx: int, y_pred_idx: int, correct: bool):

        W = float(self.max_window_size)

        if correct:

            c = y_true_idx
            self.eps_[k, c] = max(0.0, self.eps_[k, c] - 1.0)
            lam = W * math.exp(-float(self.eps_[k, c]))
            lam = max(1.0, lam)

            self.cw_[k, c] = self.cw_[k, c] * (lam - 1.0) / lam + 1.0 / lam
        else:

            for c in (y_true_idx, y_pred_idx):
                self.eps_[k, c] = float(self.eps_[k, c]) + 1.0
                lam = W * math.exp(-float(self.eps_[k, c]))
                lam = max(1.0, lam)

                self.cw_[k, c] = self.cw_[k, c] * (lam - 1.0) / lam

    def _remove_one_by_average_rank(self):

        K = len(self.models_)
        C = int(self.n_classes_)
        if K <= self.max_ensemble_size:
            return

        cw = self.cw_

        ranks = np.zeros((K, C), dtype=float)
        for c in range(C):
            order = np.argsort(-cw[:, c])

            for r, idx in enumerate(order, start=1):
                ranks[idx, c] = r

        avg_rank = ranks.mean(axis=1)
        remove_idx = int(np.argmax(avg_rank))


        self.models_.pop(remove_idx)
        self.cw_ = np.delete(self.cw_, remove_idx, axis=0)
        self.eps_ = np.delete(self.eps_, remove_idx, axis=0)

    def _maybe_create_new_model_from_cache(self):
        if len(self._buf_y) < self.chunk_size:
            return

        Xb = np.asarray(self._buf_X, dtype=float)
        yb = np.asarray(self._buf_y)

        new_model = self._clone_estimator()

        new_model.partial_fit(Xb, yb, classes=self.classes_)


        C = int(self.n_classes_)
        if self.cw_ is None:
            self.cw_ = np.ones((0, C), dtype=float)
            self.eps_ = np.zeros((0, C), dtype=float)

        self.models_.append(new_model)
        self.cw_ = np.vstack([self.cw_, np.ones((1, C), dtype=float)])
        self.eps_ = np.vstack([self.eps_, np.zeros((1, C), dtype=float)])


        self._buf_X.clear()
        self._buf_y.clear()


        while len(self.models_) > self.max_ensemble_size:
            self._remove_one_by_average_rank()


    def partial_fit(self, X, y, classes=None):

        X = self._ensure_2d(X)
        y = self._ensure_1d_y(y)

        if classes is not None:
            self.classes_ = np.asarray(classes)
        elif self.classes_ is None:
            self.classes_ = np.unique(y)

        self.n_classes_ = int(len(self.classes_))

        for i in range(X.shape[0]):
            xi = X[i].reshape(1, -1)
            yi = y[i]
            y_true_idx = int(np.where(self.classes_ == yi)[0][0])


            if len(self.models_) > 0:
                for k, model in enumerate(self.models_):

                    proba_k = self._safe_predict_proba(model, xi)[0]
                    y_pred_idx = int(np.argmax(proba_k))
                    correct = (y_pred_idx == y_true_idx)


                    self._update_weight_vector_one_instance(k, y_true_idx, y_pred_idx, correct)


                    model.partial_fit(xi, np.asarray([yi]), classes=self.classes_)


            self._buf_X.append(xi.reshape(-1))
            self._buf_y.append(int(yi) if np.issubdtype(type(yi), np.integer) else yi)


            self._maybe_create_new_model_from_cache()

        self._fitted_once = True
        return self

    def predict_proba(self, X):
        X = self._ensure_2d(X)
        if not self._fitted_once or self.classes_ is None or self.n_classes_ is None:
            raise RuntimeError("Model has not been fitted. Call partial_fit first.")

        C = int(self.n_classes_)


        if len(self.models_) == 0:
            return np.ones((X.shape[0], C), dtype=float) / float(C)


        cw = np.asarray(self.cw_, dtype=float)
        denom = cw.sum(axis=0, keepdims=True)
        denom[denom == 0.0] = 1.0
        ncw = cw / denom

        proba_ens = np.zeros((X.shape[0], C), dtype=float)
        for k, model in enumerate(self.models_):
            proba_k = self._safe_predict_proba(model, X)
            proba_ens += proba_k * ncw[k].reshape(1, -1)


        s = proba_ens.sum(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        return proba_ens / s

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
