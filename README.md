# HNTSMRG24 Challenge Submission
This repo accompanies our submission to the HNTSMRG24 challenge. The challenge paper will be made available after the final subission deadline:

Elias Tappiner, Christian Gapp, Martin Welk and Rainer Schuber (2024) CNNs to the Rescue: Head and Neck Tumor Segmentation on MRIs with staged nnU-Nets

The repository basically follows the official [nnU-Net repo](https://github.com/MIC-DKFZ/nnUNet/tree/v2.4.2) with a few adaptation to guide the patch sampling based on an additional input channel.

First make sure Torch is installed and able to access your GPUs.
Next install our Version of the nnU-Net
`pip install -e .`

Set up [environmentvariables](https://github.com/MIC-DKFZ/nnUNet/blob/v2.4.2/documentation/set_environment_variables.md) and create the dataset folder as described in the [installation instruction](https://github.com/MIC-DKFZ/nnUNet/blob/v2.4.2/documentation/dataset_format.md) of the nnU-Net repository.

## Training
The nnU-Net configuration and training for the pre-RT task can be run as follows (assuming the hntsmrg dataset id is 030):

- `nnUNetv2_extract_fingerprint -d 30 --verify_dataset_integrity`
- `nnUNetv2_plan_experiment -d 30 -pl nnUNetPlannerResEncM -gpu_memory_target 24 -overwrite_plans_name nnUNetResEncUNetMPlans_24Gb`
- `nnUNetv2_preprocess -d 30 -plans_name nnUNetResEncUNetMPlans_24Gb -c 3d_fullres -np 12 --verbose`
- `CUDA_VISIBLE_DEVICES=X nnUNetv2_train 30 3d_fullres [0,1,2,3,4,all] -p nnUNetResEncUNetMPlans_24Gb`

For the mid-RT we relay on our changes to train a second-stage network using the data of the pre-RT as additional input. The dataset of the mid-RT task requires three input modalidies:
- 0000 -> mid-RT MRI
- 0001 -> registred pre-RT MIR
- 0002 -> registred pre-RT reference (modality which is used to guide the training process)

After setting up the dataset, the nnU-Net configuration for the mid-RT task can be run as follows (assuming the hntsmrg dataset id is 031):

- `nnUNetv2_extract_fingerprint -d 30 --verify_dataset_integrity`
- `nnUNetv2_plan_experiment -d 30 -pl nnUNetPlannerResEncM -gpu_memory_target 24 -preprocessor_name DilatedPremaskAsFgPreprocessor -overwrite_plans_name nnUNetResEncUNetMPlans_24Gb_masked_oversampler`
- `nnUNetv2_preprocess -d 30 -plans_name nnUNetResEncUNetMPlans_24Gb -c 3d_fullres -np 12 --verbose`

Next, the generated `nnUNetResEncUNetMPlans_24Gb_masked_oversampler.json` file needs to be extended with the following configuration:

```
  "3d_fullres_oversample": {
      "inherits_from": "3d_fullres",
      "data_identifier": "nnUNetPlans_3d_fullres_masked_oversampler_4c",
      "dilationradius": 0,
      "channel_to_dilate": 2,
      "fg_oversampling": 1.0,
      "rem_dilchannel": false,
      "infer": false,
      "patch_skipping": true,
      "resampling_fn_data": "resample_data_or_seg_dilated_premask_to_shape",
      "resampling_fn_data_kwargs": {
          "is_seg": false,
          "order": 3,
          "order_z": 0,
          "force_separate_z": null,
          "seg_channel": 2
      }
  }
```

Finally, the mid-RT second stage network can be trained:

- `CUDA_VISIBLE_DEVICES=X nnUNetv2_train 31 3d_fullres_masked_oversampler [0,1,2,3,4,all] -p nnUNetResEncUNetMPlans_24Gb_masked_oversampler`

## Inference
In [workdir](workdir) we also provide the setup to infer from the trained models. 

For the pre-RT task, first the [checkpoint](https://drive.google.com/file/d/1PqC1a2DgOKB4aNwX-HbgC9OK6NwWLKyC/view?usp=sharing) needs to be downloaded and moved to [workdir/nnUNet_trained_models/Dataset030/nnUNetTrainer__nnUNetResEncUNetMPlans_24Gb__3d_fullres/fold_all](workdir/nnUNet_trained_models/Dataset030/nnUNetTrainer__nnUNetResEncUNetMPlans_24Gb__3d_fullres/fold_all) before running inference with:

- `CUDA_VISIBLE_DEVICES=X nnUNetv2_predict -i path/to/input_mris -o path/to/seg_resualts -d 30 -c 3d_fullres -p nnUNetResEncUNetMPlans_24Gb -f all`

Similar, for the mid-RT, the final [checkpoint](https://drive.google.com/file/d/1rG2wy1QCw4IyenEugjVk8P8NYx6uQ7Yi/view?usp=sharing) needs to be downloaded and moved to [workdir/nnUNet_trained_models/Dataset031/nnUNetTrainer__nnUNetResEncUNetMPlans_24Gb_masked_oversampler__3d_fullres_oversample/fold_all](workdir/nnUNet_trained_models/Dataset031/nnUNetTrainer__nnUNetResEncUNetMPlans_24Gb_masked_oversampler__3d_fullres_oversample/fold_all) before running inference with:

- `CUDA_VISIBLE_DEVICES=X nnUNetv2_predict -i path/to/input_mris -o path/to/seg_resualts -d 31 -c 3d_fullres_oversample -p nnUNetResEncUNetMPlans_24Gbb_masked_oversampler -f all`


<br/>
<br/>

When in doubt just reach out, we are happy to help (the repo is just set up as a quick and dirty reference to show how we got our results):-) 
