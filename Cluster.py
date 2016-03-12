import gc
import time

import numpy as np
from sklearn.cluster import DBSCAN


class Cluster():
    def __init__(self, img_data, mask):
        self.im = img_data
        self.mask = mask
        self.multiD = []

    def DBSCAN(self, start_frame, end_frame, size_threshold=75, max_dist=19, min_samp=10, dist_weight=3.0,
               color_weight=18.0, metric="euclidean", algo="auto"):
        shape = np.array(self.im).shape
        x_size = shape[2]
        y_size = shape[1]
        print("Stacking data for multi-dimensional clustering.")
        for i_frame in range(start_frame, end_frame):
            indicies = np.dstack((np.dstack(np.indices((y_size, x_size)) * dist_weight),
                                  np.full((y_size, x_size), i_frame * dist_weight, dtype="float64")))
            im_merged = np.dstack((self.im[i_frame] * color_weight, indicies))[self.mask[i_frame]]
            if i_frame == start_frame:
                self.multiD = im_merged
            else:
                self.multiD = np.concatenate((self.multiD, im_merged), 0)
        print("Finished stacking.")
        gc.collect()
        print("Starting DBSCAN with a total of {} points.".format(str(len(self.multiD))))
        oldtime = time.time()
        db = DBSCAN(eps=max_dist, min_samples=min_samp, metric=metric, algorithm=algo).fit(self.multiD)
        newtime = time.time()
        print("Clustering complete. Total time: {} sec".format(str(newtime - oldtime)))
        labels = db.labels_
        unique_labels = set(labels)
        colors = np.random.rand(len(unique_labels), 4)
        positions = self.multiD[:, 3:]
        results_container = np.zeros((end_frame - start_frame, y_size, x_size, 3), dtype="uint8")
        for k, col in zip(unique_labels, colors):
            if k == -1:
                continue
            class_member_mask = (labels == k)
            if np.sum(class_member_mask) < size_threshold:
                continue
            valid_points = positions[class_member_mask]
            for point in valid_points:
                x = int(point[0] / dist_weight)
                y = int(point[1] / dist_weight)
                z = int(point[2] / dist_weight) - start_frame
                results_container[z, x, y] = col[0:3] * 255
        return results_container
