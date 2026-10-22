"""Small shallow-tree logistic boosting model, exported for NumPy-only runtime."""
import numpy as np


def predict_tree(node, x, depth=0):
    if depth > 3:
        raise ValueError('Curved tree exceeds supported depth')
    if 'value' in node:
        value = float(node['value'])
        if not np.isfinite(value):raise ValueError('Nonfinite tree leaf')
        return np.full(len(x), value)
    feature = node['feature']; threshold = float(node['threshold'])
    if type(feature) is not int or not 0 <= feature < x.shape[1] or not np.isfinite(threshold):
        raise ValueError('Invalid tree split')
    left = x[:, feature] <= threshold
    result = np.empty(len(x))
    result[left] = predict_tree(node['left'], x[left], depth+1)
    result[~left] = predict_tree(node['right'], x[~left], depth+1)
    return result


def fit_tree(x, gradient, hessian, regularization, depth=0):
    g = gradient.sum(); h = hessian.sum()
    leaf = {'value': float(np.clip(-g/(h+regularization), -3, 3))}
    if depth >= 2 or len(x) < 12:return leaf
    best = None; best_gain = 1e-6
    for feature in range(x.shape[1]):
        order = np.argsort(x[:, feature], kind='stable')
        values = x[order, feature]; left_g = np.cumsum(gradient[order])[:-1]; left_h = np.cumsum(hessian[order])[:-1]
        gains = left_g**2/(left_h+regularization)+(g-left_g)**2/(h-left_h+regularization)-g*g/(h+regularization)
        valid = (values[:-1] < values[1:]) & (np.arange(1, len(x)) >= 6) & (np.arange(1, len(x)) <= len(x)-6)
        gains[~valid] = -np.inf; split = int(np.argmax(gains))
        if gains[split] > best_gain:
            best_gain = float(gains[split]); best = (feature, float((values[split]+values[split+1])/2))
    if best is None:return leaf
    feature, threshold = best; mask = x[:, feature] <= threshold
    return {'feature': feature, 'threshold': threshold,
            'left': fit_tree(x[mask], gradient[mask], hessian[mask], regularization, depth+1),
            'right': fit_tree(x[~mask], gradient[~mask], hessian[~mask], regularization, depth+1)}


def fit(x, y, regularization, feature_names, sample_weights=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.ndim != 2 or x.shape != (len(y), len(feature_names)) or not np.isfinite(x).all() or set(y) != {0., 1.} or regularization <= 0:
        raise ValueError('Invalid curved-tree training data')
    weights = np.where(y == 1, len(y)/(2*y.sum()), len(y)/(2*(1-y).sum()))
    if sample_weights is not None:
        supplied=np.asarray(sample_weights,float)
        if supplied.shape!=y.shape or not np.isfinite(supplied).all() or np.any(supplied<=0):
            raise ValueError('Invalid curved-tree sample weights')
        weights*=supplied
    logits = np.zeros(len(x)); trees = []; rate = .15
    for _ in range(30):
        probability = 1/(1+np.exp(-np.clip(logits, -35, 35)))
        tree = fit_tree(x, weights*(probability-y), weights*probability*(1-probability), regularization)
        logits += rate*predict_tree(tree, x); trees.append(tree)
    return {'schema_version': 1, 'kind': 'shallow_boosted_trees', 'features': feature_names,
            'trees': trees, 'learning_rate': rate, 'regularization': regularization,
            'explicit_non_table_training_events_upweighted': sample_weights is not None,
            'score_is_calibrated_probability': False}


def score_matrix(model, x):
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[1] != len(model['features']) or not np.isfinite(x).all():raise ValueError('Invalid curved-tree inputs')
    rate = float(model['learning_rate']); trees = model['trees']
    if not np.isfinite(rate) or not 0 < rate <= 1 or not 1 <= len(trees) <= 60:raise ValueError('Invalid curved-tree model')
    logits = sum((rate*predict_tree(tree, x) for tree in trees), np.zeros(len(x)))
    return 1/(1+np.exp(-np.clip(logits, -35, 35)))
