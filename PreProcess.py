import cPickle as pickle
import collections as coll
import gc
import multiprocessing as mp
import time
from os import path

import numpy as np
from scipy import ndimage as ndi

from MPfuncs import edge_func, intensity_func, iso_func, li_func, otsu_func, watershed_func, yen_func, lab_func


class PreProcess:
    def __init__(self, im_data, filepath, snapshotpath, multiprocess=True, cores="auto"):
        """Accepts im_data from NeuronStack class instance,
           multiprocess attempts to speed up certain steps,
           cores are manually detected unless given"""
        self.im = im_data
        self.imname = filepath
        self.savepath = snapshotpath
        self.multi = multiprocess
        self.threhold_method = ""
        self.cores = mp.cpu_count()
        self.intensity = []
        self.masks = []
        self.threshold_arr = []
        if isinstance(cores, int):
            self.cores = cores

    def blur(self, blur_sigma, xyz_scale=(1, 1, 1)):
        """Blurs the image, blur_sigma can be either an int
           or an list of ints, the sigmas are for x, y and z.
           xyz_scale should be the resolution in each axis"""

        if not isinstance(blur_sigma, coll.Iterable):
            blur_sigma = np.array([blur_sigma, blur_sigma, blur_sigma], dtype=np.double)
        elif isinstance(blur_sigma, (list, tuple)):
            blur_sigma = np.array(blur_sigma, dtype=np.double)
            if len(blur_sigma) != 3 or len(xyz_scale) != 3:
                raise ValueError('Array dimension was not correct')

        avg = sum(xyz_scale) / 3.0
        xyz_scale = np.array(xyz_scale, dtype=np.double) / avg
        blur_sigma /= xyz_scale
        blur_sigma = list(blur_sigma)[::-1]
        # Reverses the xyz sigmas, so it's now zyx, and matches array axis
        blur_sigma = blur_sigma + [0]
        # Adds element to indicate no blurring between colors
        print("Gaussian blur starting...")
        self.im = ndi.filters.gaussian_filter(self.im, blur_sigma)
        print("Gaussian blur finished.")
        return self.im

    def find_threshold(self, method="isodata", snapshot=True):
        """Provides a function to algorithmically find the
           threshold of an multi-page tif file, available
           methods includes (case insensitive): ISODATA, Li,
           Otsu, Yen. Returns a list of thresholds for each
           z-slice."""
        method = method.lower()
        print("Method selected: {}".format(method))
        methods = {"isodata": iso_func, "li": li_func, "otsu": otsu_func, "yen": yen_func}
        self.threhold_method = method
        if path.isfile(self.savepath + self.imname + "_" + str(method) + ".p"):
            print("Old threshold data detected. Loading from disk.")
            self.threshold_arr = pickle.load(open(self.savepath + self.imname + "_" + str(method) + ".p", "rb"))
            print("Old threshold data loaded.")
            return self.threshold_arr

        if self.multi:
            # Core path for multi-core
            print("Starting multi-core threshold finding with {} cores.".format(self.cores))
            if __name__ == 'PreProcess':
                pool = mp.Pool(self.cores)
                if len(self.intensity) == 0:
                    self.intensity = pool.map(intensity_func, self.im)
                pool.close()
                pool.join()
                pool = mp.Pool(self.cores)
                oldtime = time.time()
                thresholds = pool.map(methods[method], self.intensity)
                pool.close()
                pool.join()
                newtime = time.time()
                print("Multi-core thresholding with {}'s method finished. Total time: {} sec".format(method, str(
                    newtime - oldtime)))

        elif not self.multi:
            # Code path for non multi-core
            print("Starting single-core threshold finding.")
            if len(self.intensity) == 0:
                self.intensity = map(intensity_func, self.im)
            oldtime = time.time()
            thresholds = map(methods[method], self.intensity)
            newtime = time.time()
            print("Single-core thresholding with {}'s method finished. Total time: {} sec".format(method, str(
                newtime - oldtime)))
        if snapshot:
            print("Saving snapshot of computed thresholds...")
            filepath = self.savepath + self.imname + "_" + str(method) + ".p"
            pickle.dump(thresholds, open(filepath, "wb"))
            print("Results saved at {}".format(filepath))

        self.threshold_arr = thresholds
        return self.threshold_arr

    def sobel_watershed(self, threshold="last", snapshot=True):
        """Provides a function to isolate the neuron from
           the background.
           """
        if path.isfile(self.savepath + self.imname + "_" + self.threhold_method + "_mask.p"):
            print("Old mask data detected. Loading from disk.")
            cached_mask = pickle.load(open(self.savepath + self.imname + "_" + self.threhold_method + "_mask.p", "rb"))
            print("Old mask data loaded.")
            self.masks = cached_mask
            return cached_mask

        if threshold == "last" and len(self.threshold_arr) != 0:
            threshold = self.threshold_arr
        elif threshold == "last" and len(self.threshold_arr) == 0:
            print("Threshold has not been calculated, calculating now...")
            threshold = self.find_threshold("isodata", snapshot=True)
        elif isinstance(threshold, list):
            threshold = threshold
        else:
            raise ValueError("Threshold format not valid. It must be a list.")

        if self.multi:
            print("Starting multi-core mask generation with {} cores...".format(self.cores))
            if __name__ == 'PreProcess':
                pool = mp.Pool(self.cores)
                if len(self.intensity) == 0:
                    self.intensity = pool.map(intensity_func, self.im)
                pool.close()
                pool.join()
                oldtime = time.time()
                print("Starting edge detection...")
                pool = mp.Pool(self.cores)
                edge_map = pool.map(edge_func, self.intensity)
                pool.close()
                pool.join()
                print("Preparing for array...")
                for frame_ID in range(0, len(edge_map)):
                    edge_map[frame_ID] = [edge_map[frame_ID], self.intensity[frame_ID], threshold[frame_ID]]
                print("Starting multi-core watershed mask refinement with {} cores...".format(self.cores))
                pool = mp.Pool(self.cores)
                masks = pool.map(watershed_func, edge_map)
                pool.close()
                pool.join()
                newtime = time.time()
                print("Multi-core mask refinement finished. Total time: {} sec".format(str(newtime - oldtime)))

        elif not self.multi:
            print("Starting single-core mask generation")
            if len(self.intensity) == 0:
                self.intensity = map(intensity_func, self.im)
            oldtime = time.time()
            edge_map = map(edge_func, self.intensity)
            print("Preparing for array...")
            for frame_ID in range(0, len(edge_map)):
                edge_map[frame_ID] = [edge_map[frame_ID], self.intensity[frame_ID], self.threshold_arr[frame_ID]]
            print("Starting single-core watershed mask refinement...")
            masks = map(watershed_func, edge_map)
            newtime = time.time()
            print("Single-core mask refinement finished. Total time: {} sec".format(str(newtime - oldtime)))

        if snapshot:
            print("Saving snapshot of computed masks...")
            filepath = self.savepath + self.imname + "_" + self.threhold_method + "_mask.p"
            pickle.dump(masks, open(filepath, "wb"))
            print("Results saved at {}".format(filepath))
        self.masks = masks
        return masks

    def apply_mask(self, mask="last"):

        if mask == "last":
            mask = self.masks
        print("Starting masking on a single thread...")
        for z_slice in range(0, len(self.im)):
            im_slice = self.im[z_slice]
            mask_slice = mask[z_slice]
            im_slice[~mask_slice] = [0, 0, 0]
            self.im[z_slice] = im_slice
        print("Masking finished")
        return self.im

    def lab_mode(self):
        if self.multi:
            if __name__ == 'PreProcess':
                pool = mp.Pool(self.cores)
                print("Starting multi-core LAB conversion with {} cores.".format(self.cores))
                oldtime = time.time()
                self.im = pool.map(lab_func, self.im)
                pool.close()
                pool.join()
                newtime = time.time()
                print("LAB color space conversion finished. Total time: {} sec".format(str(newtime - oldtime)))
        elif not self.multi:
            print("Starting single-core LAB conversion.")
            oldtime = time.time()
            self.im = map(lab_func, self.im)
            newtime = time.time()
            print("LAB color space conversion finished. Total time: {} sec".format(str(newtime - oldtime)))
        return self.im

    def return_data(self):
        del self.intensity
        del self.masks
        gc.collect()
        return self.im
