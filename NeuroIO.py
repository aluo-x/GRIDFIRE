import numpy as np
from os import path
from skimage import data
from libtiff import TIFF


class NeuroIO:
    def __init__(self, im_path):
        """Accepts the path to a tif or tiff file,
           then attempts to output a numpy uint8 array
           that has the shape (z, y, x, color)"""
        if not path.isfile(im_path):
            raise IOError("File does not exist")
        if not im_path.lower().endswith(('.tif', '.tiff')):
            raise IOError("File type incorrect")

        self.path = im_path
        self.im = data.imread(im_path)
        self.im_shape = self.im.shape
        self.new_im = self.im
        color_axis = np.argmin(self.im_shape)
        if color_axis == 1:
            self.new_im = []
            print("Restacking array... current size: {}".format(str(self.im_shape)))
            for z_slice in range(self.im_shape[0]):
                for channel in range(self.im_shape[color_axis]):
                    s_data = self.im[z_slice]
                    self.new_im.append(np.dstack((s_data[0], s_data[1], s_data[2])))
            self.new_im = np.array(self.new_im, dtype="uint8")
            print("Restack completed. current size: {}".format(str(self.new_im.shape)))
        if color_axis != 1 and color_axis != 3:
            raise ValueError("File format not recognized")

    def img_data_return(self):
        """Returns the re-formatted data"""
        return (self.new_im, path.basename(self.path))

    def img_data_write(self, im_data, save_path):
        list_r = []
        list_g = []
        list_b = []
        for frame_data in im_data:
            list_r.append(frame_data[:, :, 0])
            list_g.append(frame_data[:, :, 1])
            list_b.append(frame_data[:, :, 2])
        tiff = TIFF.open(save_path + "r.tiff", "w")
        tiff.write_image(list_r)
        tiff.close()
        tiff = TIFF.open(save_path + "g.tiff", "w")
        tiff.write_image(list_g)
        tiff.close()
        tiff = TIFF.open(save_path + "b.tiff", "w")
        tiff.write_image(list_b)
        tiff.close()
