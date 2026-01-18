import copy
import numpy as np

from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.drift_detection import ADWIN

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def _safe_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp_z = np.exp(z)
    s = exp_z.sum()
    if s <= 0:
        return np.ones_like(logits) / len(logits)
    return exp_z / s


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _quantile_cutpoints(z: np.ndarray, k: int) -> np.ndarray:
    z_sorted = np.sort(z)
    cutpoints = [-np.inf]
    for i in range(1, k):
        idx = int(np.floor(i * len(z_sorted) / k))
        idx = min(max(idx, 0), len(z_sorted) - 1)
        cutpoints.append(z_sorted[idx])
    cutpoints.append(np.inf)
    return np.array(cutpoints, dtype=float)


def _assign_subspace(z_val: float, boundaries: np.ndarray) -> int:
    sid = int(np.searchsorted(boundaries[1:], z_val, side="right"))
    return min(max(sid, 0), len(boundaries) - 2)


def _pad_to_dim(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[0] == dim:
        return x
    if x.shape[0] < dim:
        return np.concatenate([x, np.zeros((dim - x.shape[0],), dtype=float)], axis=0)
    return x[:dim]


class FTRLProximal:

    def __init__(self, n_features: int, n_classes: int,
                 alpha: float = 0.1, beta: float = 1.0,
                 l1: float = 1.0, l2: float = 1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.l1 = float(l1)
        self.l2 = float(l2)

        self.n_features = int(n_features)
        self.n_classes = int(n_classes)

        self.z = np.zeros((self.n_features, self.n_classes), dtype=float)
        self.n = np.zeros((self.n_features, self.n_classes), dtype=float)

    def expand_features(self, new_n_features: int):
        new_n_features = int(new_n_features)
        if new_n_features <= self.n_features:
            return
        extra = new_n_features - self.n_features
        self.z = np.vstack([self.z, np.zeros((extra, self.n_classes), dtype=float)])
        self.n = np.vstack([self.n, np.zeros((extra, self.n_classes), dtype=float)])
        self.n_features = new_n_features

    def get_weights(self) -> np.ndarray:
        w = np.zeros_like(self.z)
        abs_z = np.abs(self.z)
        mask = abs_z > self.l1
        denom = (self.beta + np.sqrt(self.n)) / self.alpha + self.l2
        w[mask] = -(self.z[mask] - np.sign(self.z[mask]) * self.l1) / denom[mask]
        return w

    def update(self, grad_W: np.ndarray):
        w = self.get_weights()
        n_new = self.n + grad_W * grad_W
        sigma = (np.sqrt(n_new) - np.sqrt(self.n)) / self.alpha
        self.z = self.z + grad_W - sigma * w
        self.n = n_new


class _SubspaceModel:

    def __init__(self, base_estimator, base_dim: int, total_dim: int, n_classes: int, ftrl_params: dict):
        self.base = copy.deepcopy(base_estimator)
        self.base_dim = int(base_dim)

        self.residual = FTRLProximal(
            n_features=int(total_dim),
            n_classes=int(n_classes),
            alpha=ftrl_params["alpha"],
            beta=ftrl_params["beta"],
            l1=ftrl_params["l1"],
            l2=ftrl_params["l2"],
        )

        self.drift = ADWIN()
        self.base_initialized = False

    def expand_total_dim(self, new_total_dim: int):
        self.residual.expand_features(int(new_total_dim))

    def reset(self, base_estimator, total_dim: int, n_classes: int, ftrl_params: dict):
        # base_dim must NOT change here (your definition)
        self.base = copy.deepcopy(base_estimator)
        self.residual = FTRLProximal(
            n_features=int(total_dim),
            n_classes=int(n_classes),
            alpha=ftrl_params["alpha"],
            beta=ftrl_params["beta"],
            l1=ftrl_params["l1"],
            l2=ftrl_params["l2"],
        )
        self.drift = ADWIN()
        self.base_initialized = False



class SubspaceResidualTrapezoidalClassifier(BaseSKMObject, ClassifierMixin):

    def __init__(self,
                 n_subspaces: int = 5,
                 init_size: int = 500,
                 base_dim: int = None,
                 base_estimator=None,
                 window_size_L: int = 500,
                 js_threshold: float = 0.15,
                 ftrl_alpha: float = 0.1,
                 ftrl_beta: float = 1.0,
                 ftrl_l1: float = 1.0,
                 ftrl_l2: float = 1.0,
                 random_state: int = 42):
        super().__init__()
        self.n_subspaces = int(n_subspaces)
        self.init_size = int(init_size)
        self.base_dim = base_dim

        self.base_estimator = base_estimator if base_estimator is not None else HoeffdingTreeClassifier()

        self.window_size_L = int(window_size_L)
        self.js_threshold = float(js_threshold)

        self.ftrl_params = {
            "alpha": float(ftrl_alpha),
            "beta": float(ftrl_beta),
            "l1": float(ftrl_l1),
            "l2": float(ftrl_l2),
        }

        self.random_state = int(random_state)

        # runtime states
        self._classes = None
        self._n_classes = None

        self._total_dim = None
        self._proj_w = None
        self._boundaries = None

        self._subs = None

        self._init_X = []
        self._init_y = []

        self._prev_pi = None
        self._curr_counts = None
        self._curr_fill = 0

        self._window_X = []
        self._window_y = []


    def _ensure_total_dim(self, X: np.ndarray):
        if self._total_dim is None:
            self._total_dim = int(X.shape[1])
            return

        if X.shape[1] > self._total_dim:
            self._total_dim = int(X.shape[1])
            if self._subs is not None:
                for sm in self._subs:
                    sm.expand_total_dim(self._total_dim)

    def _fit_partition(self, X_init: np.ndarray, y_init: np.ndarray):

        if self.base_dim is None:
            self.base_dim = int(X_init.shape[1])

        Xb = X_init[:, :self.base_dim]

        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(Xb, y_init)

        if hasattr(lda, "scalings_") and lda.scalings_ is not None:
            w = lda.scalings_[:, 0]
        else:
            w = lda.coef_.reshape(-1)

        w = w.astype(float)
        w = w / (np.linalg.norm(w) + 1e-12)

        z = Xb @ w
        self._proj_w = w
        self._boundaries = _quantile_cutpoints(z, self.n_subspaces)

    def _route(self, x: np.ndarray) -> int:
        xb = x[:self.base_dim]
        z_val = float(np.dot(self._proj_w, xb))
        return _assign_subspace(z_val, self._boundaries)

    def _init_models(self):
        self._subs = []
        for _ in range(self.n_subspaces):
            self._subs.append(
                _SubspaceModel(
                    base_estimator=self.base_estimator,
                    base_dim=self.base_dim,
                    total_dim=self._total_dim,
                    n_classes=self._n_classes,
                    ftrl_params=self.ftrl_params
                )
            )

        self._prev_pi = None
        self._curr_counts = np.zeros((self.n_subspaces,), dtype=float)
        self._curr_fill = 0
        self._window_X, self._window_y = [], []

    def _base_proba(self, sm: _SubspaceModel, x: np.ndarray) -> np.ndarray:
        if not sm.base_initialized:
            return np.ones((self._n_classes,), dtype=float) / self._n_classes
        xb = x[:sm.base_dim].reshape(1, -1)
        try:
            p = sm.base.predict_proba(xb)[0]
            p = np.asarray(p, dtype=float)
            if p.shape[0] != self._n_classes:
                return np.ones((self._n_classes,), dtype=float) / self._n_classes
            return p
        except Exception:
            return np.ones((self._n_classes,), dtype=float) / self._n_classes

    def _residual_logits(self, sm: _SubspaceModel, x: np.ndarray) -> np.ndarray:
        x_r = x.copy()
        x_r[:sm.base_dim] = 0.0  # residual only sees new features
        W = sm.residual.get_weights()
        return (W.T @ x_r.reshape(-1, 1)).reshape(-1)

    def _predict_proba_one(self, x: np.ndarray, sid: int) -> np.ndarray:
        sm = self._subs[sid]
        p_base = self._base_proba(sm, x)
        r = self._residual_logits(sm, x)
        logits = np.log(np.clip(p_base, 1e-12, 1.0)) + r
        return _safe_softmax(logits)

    def _boundary_window_update(self, sid: int, x: np.ndarray, y=None) -> bool:
        self._curr_counts[sid] += 1.0
        self._curr_fill += 1


        self._window_X.append(np.asarray(x, dtype=float).copy())
        self._window_y.append(None if y is None else int(y))

        if self._curr_fill < self.window_size_L:
            return False

        pi_t = self._curr_counts / (self._curr_counts.sum() + 1e-12)
        drift = False
        if self._prev_pi is not None:
            js = _js_divergence(pi_t, self._prev_pi)
            if js > self.js_threshold:
                drift = True

        self._prev_pi = pi_t
        self._curr_counts = np.zeros((self.n_subspaces,), dtype=float)
        self._curr_fill = 0
        return drift

    def _update_one_labeled(self, x: np.ndarray, y: int, sid: int):
        sm = self._subs[sid]


        p = self._predict_proba_one(x, sid)
        y_pred = int(np.argmax(p))
        err = 1.0 if (y_pred != y) else 0.0
        sm.drift.add_element(err)


        p_base = self._base_proba(sm, x)
        y_onehot = np.zeros((self._n_classes,), dtype=float)
        y_onehot[y] = 1.0

        u = y_onehot - p_base
        x_r = x.copy()
        x_r[:sm.base_dim] = 0.0
        W = sm.residual.get_weights()
        r = (W.T @ x_r.reshape(-1, 1)).reshape(-1)

        diff = (r - u).reshape(1, -1)
        grad_W = x_r.reshape(-1, 1) @ diff
        sm.residual.update(grad_W)


        xb = x[:sm.base_dim].reshape(1, -1)
        if not sm.base_initialized:
            sm.base.partial_fit(xb, np.array([y], dtype=int), classes=self._classes)
            sm.base_initialized = True
        else:
            sm.base.partial_fit(xb, np.array([y], dtype=int))


        if sm.drift.detected_change():
            sm.reset(
                base_estimator=self.base_estimator,
                total_dim=self._total_dim,
                n_classes=self._n_classes,
                ftrl_params=self.ftrl_params
            )


    def _handle_boundary_rebuild(self):

        if len(self._window_X) == 0:
            return


        labeled_idx = [i for i, yy in enumerate(self._window_y) if yy is not None]
        if len(labeled_idx) < 2:
            self._window_X, self._window_y = [], []
            return

        Xw_raw = [self._window_X[i] for i in labeled_idx]
        yw = np.array([self._window_y[i] for i in labeled_idx], dtype=int)

        dims = [x.shape[0] for x in Xw_raw]
        shared_dim = int(min(dims))
        if shared_dim <= 0:
            self._window_X, self._window_y = [], []
            return


        self.base_dim = shared_dim


        Xl_shared = np.vstack([x[:shared_dim] for x in Xw_raw]).astype(float)


        self._fit_partition(Xl_shared, yw)


        self._init_models()


        for x_raw, y in zip(Xw_raw, yw):
            x_full = _pad_to_dim(x_raw, self._total_dim)   # total_dim=全局最新维度
            sid = self._route(x_full)
            self._update_one_labeled(x_full, int(y), sid)


        self._window_X, self._window_y = [], []


        self._prev_pi = None
        self._curr_counts = np.zeros((self.n_subspaces,), dtype=float)
        self._curr_fill = 0

    def partial_fit(self, X, y=None, classes=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self._ensure_total_dim(X)

        if classes is not None and self._classes is None:
            self._classes = np.asarray(classes)
            self._n_classes = int(len(self._classes))

        # -------- init stage --------
        if self._proj_w is None:
            if y is None:
                self._init_X.extend([xi for xi in X])
                return self

            y_arr = np.asarray(y, dtype=int).reshape(-1)
            for xi, yi in zip(X, y_arr):
                self._init_X.append(np.asarray(xi, dtype=float))
                self._init_y.append(int(yi))

            if len(self._init_X) < self.init_size:
                return self

            X0_raw = self._init_X[:self.init_size]
            y0 = np.array(self._init_y[:self.init_size], dtype=int)

            if self._classes is None:
                self._classes = np.unique(y0)
                self._n_classes = int(len(self._classes))

            if self.base_dim is None:
                self.base_dim = int(min([x.shape[0] for x in X0_raw]))

            X0_shared = np.vstack([x[:self.base_dim] for x in X0_raw]).astype(float)
            self._fit_partition(X0_shared, y0)

            self._init_models()

            for x_raw, yi in zip(X0_raw, y0):
                x_full = _pad_to_dim(x_raw, self._total_dim)
                sid = self._route(x_full)
                self._update_one_labeled(x_full, int(yi), sid)

            self._init_X, self._init_y = [], []
            return self


        if y is None:

            for xi in X:
                xi_raw = np.asarray(xi, dtype=float)
                xi_full = _pad_to_dim(xi_raw, self._total_dim)
                sid = self._route(xi_full)

                drift = self._boundary_window_update(sid, xi_raw, y=None)
                if drift:
                    self._handle_boundary_rebuild()
            return self

        y_arr = np.asarray(y, dtype=int).reshape(-1)
        for xi, yi in zip(X, y_arr):
            xi_raw = np.asarray(xi, dtype=float)
            xi_full = _pad_to_dim(xi_raw, self._total_dim)
            sid = self._route(xi_full)

            drift = self._boundary_window_update(sid, xi_raw, y=int(yi))
            if drift:
                self._handle_boundary_rebuild()
                sid = self._route(xi_full)

            self._update_one_labeled(xi_full, int(yi), sid)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self._classes is None:
            return np.zeros((X.shape[0], 0), dtype=float)

        if self._proj_w is None:
            return np.ones((X.shape[0], self._n_classes), dtype=float) / self._n_classes

        self._ensure_total_dim(X)

        out = np.zeros((X.shape[0], self._n_classes), dtype=float)
        for i, xi in enumerate(X):
            xi_full = _pad_to_dim(np.asarray(xi, dtype=float), self._total_dim)
            sid = self._route(xi_full)
            out[i] = self._predict_proba_one(xi_full, sid)
        return out

    def predict(self, X):
        P = self.predict_proba(X)
        if P.shape[1] == 0:
            return np.array([], dtype=int)
        idx = np.argmax(P, axis=1)
        return self._classes[idx]

    def reset(self):
        self._classes = None
        self._n_classes = None
        self._total_dim = None
        self._proj_w = None
        self._boundaries = None
        self._subs = None
        self._init_X, self._init_y = [], []
        self._prev_pi = None
        self._curr_counts = None
        self._curr_fill = 0
        self._window_X, self._window_y = [], []
        return self


