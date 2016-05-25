import gc
import itertools
import multiprocessing as mp
import time

import numpy as np

from MPfuncs import dbscan_func
from SetMerge import label_merge, label_overlap


class Cluster:
    def __init__(self, img_data, mask):
        self.im = img_data
        self.mask = mask
        self.multiD = []
        shape = np.array(self.im).shape
        self.x_size = shape[2]
        self.y_size = shape[1]

    def img2array(self, start_frame, end_frame, dist_weight=3.0, color_weight=18.0):
        x_size = self.x_size
        y_size = self.y_size
        print("Stacking data for multi-dimensional clustering.")
        for i_frame in range(start_frame, end_frame):
            indices = np.dstack((np.dstack(np.indices((y_size, x_size)) * dist_weight),
                                 np.full((y_size, x_size), i_frame * dist_weight, dtype="float64")))
            im_merged = np.dstack((self.im[i_frame] * color_weight, indices))[self.mask[i_frame]]
            if i_frame == start_frame:
                self.multiD = im_merged
            else:
                self.multiD = np.concatenate((self.multiD, im_merged), 0)
        print("Finished stacking.")
        return self.multiD

    def split(self, num_slices, dist_weight, max_dist, axis="auto"):
        point_arr = self.multiD
        x_size = self.x_size * dist_weight
        y_size = self.y_size * dist_weight
        size = [0, 0, 0, y_size, x_size, 0]
        # order: [l, a, b, y, x, z] or [r, g, b, y, x, z]
        real_axis = axis
        if axis == "auto":
            if x_size > y_size:
                real_axis = 4
                print("Slicing along the x axis")
            elif x_size <= y_size:
                real_axis = 3
                print("Slicing along the y axis")
        overlap_dist = float(max_dist * 2.0)
        non_overlap = (size[real_axis] - overlap_dist) / float(num_slices)
        slice_container = {}
        for i_slice in range(0, num_slices):
            start = i_slice * non_overlap
            end = (i_slice + 1) * non_overlap + overlap_dist
            slice_container[i_slice] = point_arr[
                np.all([point_arr[:, real_axis] >= start - 1, point_arr[:, real_axis] <= end + 1],
                       axis=0)]  # .append(i_slice)
        return slice_container

    def super_pixel(self, start_frame, end_frame, size_threshold=75, max_dist=19, min_samp=10, dist_weight=3.0,
                    color_weight=18.0, metric="euclidean", algo="auto", multiprocess=True, num_cores="auto",
                    num_slices=4):
        self.img2array(start_frame, end_frame, dist_weight, color_weight)
        slices = self.split(num_slices, dist_weight, max_dist, axis="auto")
        slice_container = {}
        for i_slice in slices.keys():
            slice_container[i_slice] = [slices[i_slice]]
            slice_container[i_slice].append(max_dist)
            slice_container[i_slice].append(min_samp)
            slice_container[i_slice].append(metric)
            slice_container[i_slice].append(algo)
            slice_container[i_slice].append(i_slice)
        slice_container = slice_container.values()
        x_size = self.x_size
        y_size = self.y_size
        gc.collect()
        if multiprocess:
            if __name__ == 'Cluster':
                cores = mp.cpu_count()
                if isinstance(num_cores, int):
                    cores = num_cores
                pool = mp.Pool(int(cores / 2))
                oldtime = time.time()
                print("Starting DBSCAN with {} cores and {} slices.".format(int(cores / 2), num_slices))
                cluster_results = pool.map(dbscan_func, slice_container)
                pool.close()
                pool.join()
                newtime = time.time()
                print("Clustering complete. Total time: {} sec".format(str(newtime - oldtime)))
        gc.collect()

        mask_container = {}
        all_labels = []
        offset_container = []
        offset = 0
        for i_slice in range(0, num_slices):
            if i_slice in mask_container.keys():
                continue
            offset_container.append(offset)
            mask_container[i_slice] = {}
            core_samples_mask = np.zeros_like(cluster_results[i_slice][1], dtype=bool)
            core_samples_mask[cluster_results[i_slice][0]] = True
            label_container = cluster_results[i_slice][1] + offset
            slice_unique_labels = set(label_container)
            for k in slice_unique_labels:
                if k == offset - 1:
                    continue
                all_labels.append(k)
                slice_class_member_mask = (label_container == k)
                mask_container[i_slice][k] = slices[i_slice][
                    np.logical_and(slice_class_member_mask, core_samples_mask)]
            offset += len(slice_unique_labels)

        grouping_container = {}
        tupled_mask_container = {}
        for i_slice in range(0, num_slices):
            tupled_mask_container[i_slice] = {}
            for label in mask_container[i_slice].keys():
                tupled_mask_container[i_slice][label] = map(tuple, mask_container[i_slice][label])
        for i_slice in range(0, num_slices - 1):
            print("{} labels to check in this slice.".format(len(mask_container[i_slice].keys())))
            for label1 in mask_container[i_slice].keys():
                print("checking label {}".format(str(label1)))
                if label1 not in grouping_container.keys():
                    grouping_container[label1] = []
                for label2 in mask_container[i_slice + 1].keys():
                    if label2 not in grouping_container.keys():
                        grouping_container[label2] = []
                    if label_overlap(tupled_mask_container[i_slice][label1],
                                     tupled_mask_container[i_slice + 1][label2]):
                        grouping_container[label1].append(label2)

        cluster_container = {}
        merged_labels = label_merge(grouping_container)
        flattened_merged = list(itertools.chain.from_iterable(merged_labels.values()))
        for label in all_labels:
            if label not in flattened_merged:
                merged_labels[label] = [label]

        i = 0
        for same_cluster in merged_labels.values():
            valid_points = []
            for label_num in same_cluster:
                for i_slice in range(0, num_slices):
                    if label_num in mask_container[i_slice].keys():
                        valid_points.extend(mask_container[i_slice][label_num])
            valid_points = np.array(valid_points)
            cluster_container[i] = valid_points[:, 3:]
            i += 1
        results_container = np.zeros((end_frame - start_frame, y_size, x_size, 3), dtype="uint8")
        for k in cluster_container.keys():
            if len(cluster_container[k]) < size_threshold:
                continue
            labeled_points = cluster_container[k]
            color = np.random.rand(3) * 255
            for point in labeled_points:
                y = int(point[0] / dist_weight)
                x = int(point[1] / dist_weight)
                z = int(point[2] / dist_weight) - start_frame
                results_container[z, y, x] = color
        return results_container
