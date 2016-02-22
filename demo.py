if __name__ == '__main__':
    from NeuroIO import NeuroIO
    from PreProcess import PreProcess
    from Cluster import Cluster

    neuroread = NeuroIO(r"C:\Composite-1.tif")
    img_data = neuroread.img_data_return()[0]
    img_path = neuroread.img_data_return()[1]
    pre_processed_data = PreProcess(im_data=img_data, img_path=img_path, snapshotpath=r"C:\UROP\\", multiprocess=True,
                                    cores="auto")
    pre_processed_data.blur(blur_sigma=0.5, xyz_scale=(1, 1, 1))
    pre_processed_data.find_threshold(method="isodata", snapshot=True)
    refined_mask = pre_processed_data.sobel_watershed(threshold="last", snapshot=True)
    pre_processed_data.lab_mode()
    img_lab = pre_processed_data.return_data()
    segment = Cluster(img_data=img_lab, mask=refined_mask)
    cluster_results = segment.DBSCAN(start_frame=0, end_frame=100, size_threshold=75, max_dist=19, min_samp=10,
                                     dist_weight=3.0,
                                     color_weight=18.0, metric="euclidean", algo="ball_tree")
    neuroread.img_data_write(cluster_results, "C:\\")
