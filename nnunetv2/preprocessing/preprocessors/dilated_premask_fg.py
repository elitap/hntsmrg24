#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import shutil
from time import sleep
from typing import Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

from scipy.ndimage import generate_binary_structure, binary_dilation
from threading import Lock


class DilatedPremaskAsFgPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        # self.dilate_mask_lock = Lock()
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # print(properties)
        print(f"{configuration_manager.channel_to_dilate}")
        if configuration_manager.channel_to_dilate is not None:
            foreground_labels = plans_manager.get_label_manager(dataset_json).foreground_labels
            fg_mask = data[configuration_manager.channel_to_dilate].astype(np.uint8, copy=True)

            print("unique elements with count in fg_mask before dilation preprocessor:", np.unique(fg_mask, return_counts=True))

            fg_mask = self.dilate_mask(fg_mask, len(configuration_manager.spacing), foreground_labels, configuration_manager.dilationradius)

            print("unique elements with count in fg_mask after dilation preprocessor:", np.unique(fg_mask, return_counts=True))
            properties['class_locations'] = self._sample_foreground_locations(np.expand_dims(fg_mask, axis=0), foreground_labels,
                                                                            verbose=self.verbose)

            # remove we dont want to train with it!!!
            # eli was here remove this line for inference because we need it to only infer on the foreground 
            print(f"{configuration_manager.rem_dilchannel=}")
            if configuration_manager.rem_dilchannel and configuration_manager.infer == False:
                print("removing dilated channel ")
                data = np.delete(data, configuration_manager.channel_to_dilate, axis=0)
            else:
                print("keeping dilated channel")
        # print('data props after resampling', properties)

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            if configuration_manager.channel_to_dilate is None:
                # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
                # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
                # LabelManager is pretty light computation-wise.
                label_manager = plans_manager.get_label_manager(dataset_json)
                collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                    else label_manager.foreground_labels

                # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
                # collect samples uniformly from all classes (incl background)
                if label_manager.has_ignore_label:
                    collect_for_this.append(label_manager.all_labels)

                # no need to filter background in regions because it is already filtered in handle_labels
                # print(all_labels, regions)
                properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                    verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg
    
    
    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        print('processing output_filename_truncated', output_filename_truncated)
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        print('dtypes', data.dtype, seg.dtype)
        print("as unzipping is a bitch we save without zipping maybe that helps")
        # np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        np.save(output_filename_truncated + '.npy', data)
        np.save(output_filename_truncated + '_seg.npy', seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    
    def dilate_mask(self, mask: np.ndarray, dimension: int, foreground_labels: Union[List[int], List[Tuple[int, ...]]], radius: int):

        # self.dilate_mask_lock.acquire()
        # try:

        if radius < 1:
            return mask

        struct = generate_binary_structure(rank=dimension, connectivity=3)

        for label in foreground_labels:
            dil_mask = mask == label
            dil_mask = binary_dilation(dil_mask, structure=struct, iterations=radius)
            mask[dil_mask] = label
        # finally:
        #     self.dilate_mask_lock.release()
        
        return mask




def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = '/home/nnUnet_resenc/work_dir/nnUNet_preprocessed/Dataset031_HNTSMRG24_mid/nnUNetResEncUNetMPlans_24Gb_masked_oversampler.json'
    dataset_json_file = '/home/nnUnet_resenc/work_dir/nnUNet_preprocessed/Dataset031_HNTSMRG24_mid/dataset.json'
    input_images = ['/home/nnUnet_resenc/work_dir/nnUNet_raw_data_base/Dataset031_HNTSMRG24_mid/imagesTr/176_0000.nii.gz', '/home/nnUnet_resenc/work_dir/nnUNet_raw_data_base/Dataset031_HNTSMRG24_mid/imagesTr/176_0001.nii.gz', '/home/nnUnet_resenc/work_dir/nnUNet_raw_data_base/Dataset031_HNTSMRG24_mid/imagesTr/176_0002.nii.gz']  # if you only have one channel, you still need a list: ['case000_0000.nii.gz']
    seg_file = '/home/nnUnet_resenc/work_dir/nnUNet_raw_data_base/Dataset031_HNTSMRG24_mid/labelsTr/176.nii.gz'

    output_file_truncated = "work_dir/nnUNet_preprocessed/Dataset031_HNTSMRG24_mid/nnUNetPlans_3d_fullres_masked_oversampler/176"

 # 96, 165, 18, 176 , 21, 77


    configuration = '3d_fullres_masked_oversampler'
    pp = DilatedPremaskAsFgPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    # data, _, properties = pp.run_case(input_images, seg_file=None, plans_manager=plans_manager,
    #                                 configuration_manager=plans_manager.get_configuration(configuration),
    #                                 dataset_json=dataset_json_file)
    pp.run_case_save(output_file_truncated, input_images, seg_file=seg_file, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)
    
    # ret = pp.run(31, '3d_fullres_masked_oversampler', 'nnUNetResEncUNetMPlans_24Gb_masked_oversampler', 8)

    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return #data


if __name__ == '__main__':
    example_test_case_preprocessing()
    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
