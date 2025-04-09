import math
import os
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce

import cv2
import numpy as np
import psutil
import torch.utils.data as data
from inference.dataloader.infer_loader import SerializeFileList
from inference.misc.utils import log_info

from inference.infer import base


# Feature toggle
SCANNER_DRIVER = os.getenv('SCANNER_DRIVER', default='default')
ENABLE_JPG_CONVERTION = SCANNER_DRIVER == 'canon'
ENABLE_COLOR_SHIFTING = SCANNER_DRIVER == 'default'

####
def _prepare_patching(img, window_size, mask_size, return_src_top_corner=False):
    """Prepare patch information for tile processing.

    Args:
        img: original input image
        window_size: input patch size
        mask_size: output patch size
        return_src_top_corner: whether to return coordiante information for top left corner of img

    """

    win_size = window_size
    msk_size = step_size = mask_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, step_size)
    last_w, _ = get_last_steps(im_w, msk_size, step_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    # generating subpatches index from orginal
    coord_y = np.arange(0, last_h, step_size, dtype=np.int32)
    coord_x = np.arange(0, last_w, step_size, dtype=np.int32)
    row_idx = np.arange(0, coord_y.shape[0], dtype=np.int32)
    col_idx = np.arange(0, coord_x.shape[0], dtype=np.int32)
    coord_y, coord_x = np.meshgrid(coord_y, coord_x)
    row_idx, col_idx = np.meshgrid(row_idx, col_idx)
    coord_y = coord_y.flatten()
    coord_x = coord_x.flatten()
    row_idx = row_idx.flatten()
    col_idx = col_idx.flatten()
    #
    patch_info = np.stack([coord_y, coord_x, row_idx, col_idx], axis=-1)
    if not return_src_top_corner:
        return img, patch_info
    else:
        return img, patch_info, [padt, padl]


####
def _post_process_patches(
    post_proc_func,
    post_proc_kwargs,
    patch_info,
    image_info,  # overlay_kwargs,
):
    """Apply post processing to patches.

    Args:
        post_proc_func: post processing function to use
        post_proc_kwargs: keyword arguments used in post processing function
        patch_info: patch data and associated information
        image_info: input image data and associated information
        overlay_kwargs: overlay keyword arguments

    """
    # re-assemble the prediction, sort according to the patch location within the original image
    patch_info = sorted(patch_info, key=lambda x: [x[0][0], x[0][1]])
    patch_info, patch_data = zip(*patch_info)

    src_shape = image_info["src_shape"]
    src_image = image_info["src_image"]

    patch_shape = np.squeeze(patch_data[0]).shape
    ch = 1 if len(patch_shape) == 2 else patch_shape[-1]
    axes = [0, 2, 1, 3, 4] if ch != 1 else [0, 2, 1, 3]

    nr_row = max([x[2] for x in patch_info]) + 1
    nr_col = max([x[3] for x in patch_info]) + 1
    pred_map = np.concatenate(patch_data, axis=0)
    pred_map = np.reshape(pred_map, (nr_row, nr_col) + patch_shape)
    pred_map = np.transpose(pred_map, axes)
    pred_map = np.reshape(
        pred_map, (patch_shape[0] * nr_row, patch_shape[1] * nr_col, ch)
    )
    # crop back to original shape
    pred_map = np.squeeze(pred_map[: src_shape[0], : src_shape[1]])

    # * Implicit protocol
    # * a prediction map with instance of ID 1-N
    # * and a dict contain the instance info, access via its ID
    # * each instance may have type
    pred_inst, pred_type, pred_inst_centroid = post_proc_func(
        pred_map, **post_proc_kwargs
    )

    # overlaid_img = visualize_instances_dict(
    #     src_image.copy(), inst_info_dict, **overlay_kwargs
    # )

    return (
        image_info["name"],
        pred_map,
        pred_inst,
        pred_type,
        pred_inst_centroid,
    )  # , overlaid_img


class InferManager(base.InferManager):
    """Run inference on tiles."""

    ####
    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        assert self.mem_usage < 1.0 and self.mem_usage > 0.0

        # * depend on the number of samples and their size, this may be less efficient
        def patterning(x): return re.sub("([\[\]])", "[\\1]", x)
        assert os.path.isfile(
            self.file_path), "Not Detected Any Files From Path"
        file_path_list = [self.file_path]

        def detach_items_of_uid(items_list, uid, nr_expected_items):
            item_counter = 0
            detached_items_list = []
            remained_items_list = []
            while True:
                pinfo, pdata = items_list.pop(0)
                pinfo = np.squeeze(pinfo)
                if pinfo[-1] == uid:
                    detached_items_list.append([pinfo, pdata])
                    item_counter += 1
                else:
                    remained_items_list.append([pinfo, pdata])
                if item_counter == nr_expected_items:
                    break
            # do this to ensure the ordering
            remained_items_list = remained_items_list + items_list
            return detached_items_list, remained_items_list

        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        while len(file_path_list) > 0:

            hardware_stats = psutil.virtual_memory()
            available_ram = getattr(hardware_stats, "available")
            available_ram = int(available_ram * self.mem_usage)
            # available_ram >> 20 for MB, >> 30 for GB

            # TODO: this portion looks clunky but seems hard to detach into separate func

            # * caching N-files into memory such that their expected (total) memory usage
            # * does not exceed the designated percentage of currently available memory
            # * the expected memory is a factor w.r.t original input file size and
            # * must be manually provided
            file_idx = 0
            use_path_list = []
            cache_image_list = []
            cache_patch_info_list = []
            cache_image_info_list = []
            while len(file_path_list) > 0:
                file_path = file_path_list.pop(0)

                img = cv2.imread(file_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[0] == 7015 and img.shape[1] == 4960: 
                    img = img[:7015, :4960]
                    img = cv2.resize(img, (2480, 3508),
                                    interpolation=cv2.INTER_AREA)

                # shift color for ubuntu
                if ENABLE_COLOR_SHIFTING:
                    lab_jpg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    dif = int(np.mean(lab_jpg[..., 2]) - 125)
                    lab_jpg[..., 2] = lab_jpg[..., 2] - dif
                    img = cv2.cvtColor(lab_jpg, cv2.COLOR_LAB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                base_img = img.copy()
                src_shape = img.shape

                img, patch_info, top_corner = _prepare_patching(
                    img, self.patch_input_shape, self.patch_output_shape, True
                )
                self_idx = np.full(
                    patch_info.shape[0], file_idx, dtype=np.int32)
                patch_info = np.concatenate(
                    [patch_info, self_idx[:, None]], axis=-1)
                # ? may be expensive op
                patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
                patch_info = [np.squeeze(p) for p in patch_info]

                # * this factor=5 is only applicable for HoVerNet
                expected_usage = sys.getsizeof(img) * 5
                available_ram -= expected_usage
                if available_ram < 0:
                    break

                file_idx += 1
                # if file_idx == 4: break
                use_path_list.append(file_path)
                cache_image_list.append(img)
                cache_patch_info_list.extend(patch_info)
                # TODO: refactor to explicit protocol
                cache_image_info_list.append(
                    [src_shape, len(patch_info), top_corner])

            # * apply neural net on cached data
            dataset = SerializeFileList(
                cache_image_list, cache_patch_info_list, self.patch_input_shape
            )

            dataloader = data.DataLoader(
                dataset,
                num_workers=self.nr_inference_workers,
                batch_size=self.batch_size,
                drop_last=False,
            )

            accumulated_patch_output = []
            for batch_idx, batch_data in enumerate(dataloader):
                sample_data_list, sample_info_list = batch_data
                sample_output_list = self.run_step(sample_data_list)
                sample_info_list = sample_info_list.numpy()

                # edit for 512 256 164
                temp_sample_output_list = sample_output_list[:][...,
                                                                :][:, 1:-1, 1:-1]
                temp = []
                for i in range(temp_sample_output_list.shape[0]):
                    temp.append(
                        cv2.resize(
                            temp_sample_output_list[i],
                            (164, 164),
                            interpolation=cv2.INTER_AREA,
                        )
                    )
                sample_output_list = np.array(temp)

                curr_batch_size = sample_output_list.shape[0]
                sample_output_list = np.split(
                    sample_output_list, curr_batch_size, axis=0
                )
                sample_info_list = np.split(
                    sample_info_list, curr_batch_size, axis=0)
                sample_output_list = list(
                    zip(sample_info_list, sample_output_list))
                accumulated_patch_output.extend(sample_output_list)

            # * parallely assemble the processed cache data for each file if possible
            future_list = []
            for file_idx, file_path in enumerate(use_path_list):
                image_info = cache_image_info_list[file_idx]
                file_ouput_data, accumulated_patch_output = detach_items_of_uid(
                    accumulated_patch_output, file_idx, image_info[1]
                )

                # * detach this into func and multiproc dispatch it
                # src top left corner within padded image
                src_pos = image_info[2]
                src_image = cache_image_list[file_idx]
                src_image = src_image[
                    src_pos[0]: src_pos[0] + image_info[0][0],
                    src_pos[1]: src_pos[1] + image_info[0][1],
                ]

                base_name = pathlib.Path(file_path).stem
                file_info = {
                    "src_shape": image_info[0],
                    "src_image": src_image,
                    "name": base_name,
                }

                post_proc_kwargs = {
                    "nr_types": self.nr_types,
                    "return_centroids": True,
                }  # dynamicalize this

                func_args = (
                    self.post_proc_func,
                    post_proc_kwargs,
                    file_ouput_data,
                    file_info,
                )

                # dispatch for parallel post-processing
                if proc_pool is not None:
                    proc_future = proc_pool.submit(
                        _post_process_patches, *func_args)
                    # ! manually poll future and call callback later as there is no guarantee
                    # ! that the callback is called from main thread
                    future_list.append(proc_future)
            if proc_pool is not None:
                # loop over all to check state a.k.a polling
                for future in as_completed(future_list):
                    # TODO: way to retrieve which file crashed ?
                    # ! silent crash, cancel all and raise error
                    if future.exception() is not None:
                        log_info("Silent Crash")
                        # ! cancel somehow leads to cascade error later
                        # ! so just poll it then crash once all future
                        # ! acquired for now
                        # for future in future_list:
                        #     future.cancel()
                        # break
                    else:
                        (
                            img_name,
                            pred_map,
                            pred_inst,
                            pred_type,
                            pred_inst_centroid,
                        ) = future.result()
                        #     np.save('temp/pred_inst_'+img_name+'.npy',pred_inst)
                        #     np.save('temp/pred_type_'+img_name+'.npy',pred_type)
                        #     np.save('temp/pred_centroid_'+img_name+'.npy',pred_inst_centroid)
                        #     np.save('temp/test_'+img_name+'.npy',img)

                        log_info("Done Assembling %s" % img_name)
        return pred_inst, pred_type, pred_inst_centroid, base_img


def run_hover(img_path=None, model_path=None, img_size="2480,3508", model_type=None):
    nr_types = 7
    batch_size = 16
    nr_inference_workers = 8
    nr_post_proc_workers = 16
    mem_usage = 0.9
    model_mode = "fast"

    method_args = {
        "method": {
            "model_args": {
                "nr_types": nr_types,
                "mode": model_mode,
            },
            "model_path": model_path,
        },
    }

    # ***
    run_args = {
        "batch_size": int(batch_size),
        "nr_inference_workers": int(nr_inference_workers),
        "nr_post_proc_workers": int(nr_post_proc_workers),
    }

    if model_mode == "fast":
        run_args["patch_input_shape"] = 256
        run_args["patch_output_shape"] = 164
    else:
        run_args["patch_input_shape"] = 270
        run_args["patch_output_shape"] = 80

    run_args.update(
        {
            "file_path": img_path,
            "mem_usage": mem_usage,
        }
    )

    infer = InferManager(**method_args)
    return infer.process_file_list(run_args)
