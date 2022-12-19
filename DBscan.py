import numpy as np

from sklearn import datasets
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt


class CorePoint:
    def __init__(self, feature_index, vector, broadcast_complete, label):
        self.feature_index = feature_index
        self.vector = vector
        self.broadcast_complete = broadcast_complete
        self.label = label

    def __str__(self):
        return f"feature_index: {self.feature_index} - " \
               f"vec: {self.vector} - broadcast_complete: {self.broadcast_complete} - label: {self.label} "

    def __repr__(self):
        return self.__str__()


def search_core_points(_start_point, _filtered_features, add_label, _excluded_indices=list(),
                       e=0.5, min_points=4, append_start_point=True):
    features_excluded_start_point = [f for f in _filtered_features if f[0] != _start_point.feature_index]
    _core_points = list()

    for compare_feature in features_excluded_start_point:
        feature_vec = compare_feature[1]
        vec = [feature_vec[0] - _start_point.vector[0], feature_vec[1] - _start_point.vector[1]]
        length = math.sqrt(vec[0] ** 2 + vec[1] ** 2)

        if length < e and compare_feature[0] not in _excluded_indices:
            _core_points.append(CorePoint(feature_index=compare_feature[0], vector=compare_feature[1],
                                          broadcast_complete=False, label=add_label))

    if append_start_point:
        _core_points.append(_start_point)

    if len(_core_points) >= min_points:
        return _core_points
    else:
        return list()


def execute_first_broadcast(_features, add_label, start_label, tries=10, e=0.5, min_points=4):
    choices = [f[0] for f in _features]
    indices_to_try = np.random.choice(a=choices, size=tries)

    for current_index in indices_to_try:
        feature_vec = next((f[1] for f in _features if f[0] == current_index), None)
        start_point = CorePoint(feature_index=current_index, vector=feature_vec,
                                broadcast_complete=True, label=start_label)

        _core_points = search_core_points(_start_point=start_point, _filtered_features=_features,
                                          add_label=add_label, e=e, min_points=min_points)

        if len(_core_points) > 0:
            return _core_points

    return list()


def broadcast_core_points(_core_points, _features, add_label, e=0.5, min_points=4):
    core_points_copied = _core_points.copy()

    while not all([b.broadcast_complete for b in core_points_copied]):
        new_start_point = next((s for s in core_points_copied if not s.broadcast_complete), None)

        if new_start_point is not None:
            new_start_point.broadcast_complete = True

            excluded_indices = [c.feature_index for c in core_points_copied]
            next_core_points = search_core_points(_start_point=new_start_point, _filtered_features=_features,
                                                  add_label=add_label, _excluded_indices=excluded_indices,
                                                  append_start_point=False, e=e, min_points=min_points)

            for n in next_core_points:
                core_points_copied.append(n)

    return core_points_copied


def cluster_dbscan(_features, clusters=["red", "lightgreen", "lightblue"], e=0.5, min_points=4,
                   start_label="black", outliers="gray"):
    copied_features = [(idx, vec) for idx, vec in enumerate(_features)]
    completed_broadcast = list()

    for current_cluster in clusters:
        if len(copied_features) == 0:
            break

        core_points = execute_first_broadcast(_features=copied_features, add_label=current_cluster,
                                              start_label=start_label, e=e, min_points=min_points)

        if len(core_points) > 0:
            result_core_points = broadcast_core_points(_core_points=core_points, _features=copied_features,
                                                       add_label=current_cluster, e=e, min_points=min_points)

            for r in result_core_points:
                completed_broadcast.append(r)

            exclude_indices = [c.feature_index for c in completed_broadcast]
            copied_features = [cf for cf in copied_features if cf[0] not in exclude_indices]


    # print("----------ended------------------")
    # print(len(completed_broadcast))
    # print(completed_broadcast)
    # print(len(copied_features))
    # print(copied_features)

    all_points = [(bc.feature_index, bc.vector[0], bc.vector[1], bc.label) for bc in completed_broadcast]

    for cf in copied_features:
        all_points.append((cf[0], cf[1][0], cf[1][1], outliers))

    # all_points.sort(key=lambda tup: tup[0])
    # print(len(all_points))
    # print(all_points)

    return all_points


def cluster_features(_features, e=1.2, min_points=3, plot=True):
    x = [x[0] for x in _features]
    y = [y[1] for y in _features]

    clustered_points = cluster_dbscan(_features=_features, e=e, min_points=min_points)

    x2 = [c[1] for c in clustered_points]
    y2 = [c[2] for c in clustered_points]
    c2 = [c[3] for c in clustered_points]

    if plot:
        plt.scatter(x=x2, y=y2, c=c2)
        plt.title("dbscan")
        plt.show()

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(_features)

    if plot:
        plt.scatter(x, y, c=kmeans.labels_)
        plt.title("kmeans")
        plt.show()


if __name__ == '__main__':
    features, labels = datasets.make_blobs(n_samples=300, random_state=42)
    cluster_features(features)

    features, labels = datasets.make_moons(n_samples=300)
    cluster_features(features, e=0.5, min_points=2)

    features, labels = datasets.make_circles(n_samples=300)
    cluster_features(features, e=0.15, min_points=3)
