"""
Target-aware Pipeline and FeatureUnion for scikit-learn.

Propagates (X, y) through transformers instead of X only.
"""

from collections import defaultdict
import sys

import numpy as np
import six
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.metaestimators import available_if

from .sklearnbaseyt import TransformerMixinYt

# Compatibility shim for sklearn.externals.six
sys.modules['sklearn.externals.six'] = six


class PipelineYt(Pipeline):
    def _fit(self, X, y=None, **fit_params):
        self._validate_steps()
        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        Yt = y
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            elif hasattr(transform, "fit_transform"):
                Xt, Yt = transform.fit_transform(Xt, Yt, **fit_params_steps[name])
            else:
                Xt, Yt = transform.fit(Xt, Yt, **fit_params_steps[name]) \
                              .transform(Xt, Yt)
        if self._final_estimator is None:
            return Xt, Yt, {}
        return Xt, Yt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        Xt, Yt, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, Yt, **fit_params)
        return self

    def _final_estimator_has(attr):
        def check(self):
            return hasattr(self._final_estimator, attr)
        return check

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, y=None):
        Xt = X
        Yt = y
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt, Yt = transform.transform(Xt, Yt)
        return self.steps[-1][-1].predict(Xt), Yt

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, y=None):
        Xt = X
        Yt = y
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt, Yt = transform.transform(Xt, Yt)
        return self.steps[-1][-1].predict_proba(Xt), Yt

    @property
    def transform(self):
        # _final_estimator is None or has transform, otherwise attribute error
        if self._final_estimator is not None:
            self._final_estimator.transform
        return self._transform

    def _transform(self, X, y=None):
        Xt = X
        Yt = y
        for name, transform in self.steps:
            if transform is not None:
                Xt, Yt = transform.transform(Xt, Yt)
        return Xt, Yt

    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None):
        Xt = X
        Yt = y
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt, Yt = transform.transform(Xt, Yt)
        return self.steps[-1][-1].score(Xt, Yt)


def _name_estimators(estimators):
    """Generate names for estimators."""

    names = [type(estimator).__name__.lower() for estimator in estimators]
    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1

    for k, v in list(six.iteritems(namecount)):
        if v == 1:
            del namecount[k]

    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += "-%d" % namecount[name]
            namecount[name] -= 1

    return list(zip(names, estimators))

def make_pipeline_yt(*steps):
    return PipelineYt(_name_estimators(steps))


def _fit_one_transformer(transformer, X, y):
    return transformer.fit(X, y)


def _transform_one(transformer, weight, X, y):
    X_out, y_out = transformer.transform(X, y)
    if weight is None:
        return X_out, y_out
    return X_out * weight, y_out


def _fit_transform_one(transformer, weight, X, y, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        X_out, y_out = transformer.fit_transform(X, y, **fit_params)
    else:
        X_out, y_out = transformer.fit(X, y, **fit_params).transform(X, y)
    if weight is None:
        return X_out, y_out, transformer
    return X_out * weight, y_out, transformer


class FeatureUnionYt(FeatureUnion, TransformerMixinYt):
    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            return np.zeros((X.shape[0], 0)), y
        Xs, Ys, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            X_combined = sparse.hstack(Xs).tocsr()
        else:
            X_combined = np.hstack(Xs)
        return X_combined, Ys[0] if Ys else y

    def transform(self, X, y):
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X, y)
            for name, trans, weight in self._iter())
        if not result:
            return np.zeros((X.shape[0], 0)), y
        Xs, Ys = zip(*result)
        if any(sparse.issparse(f) for f in Xs):
            X_combined = sparse.hstack(Xs).tocsr()
        else:
            X_combined = np.hstack(Xs)
        return X_combined, Ys[0] if Ys else y


def make_union_yt(*transformers):
    return FeatureUnionYt(_name_estimators(transformers))
