# hntsmrg24
This repo accompanies our submission to the HNTSMRG24 challenge. The challenge paper will be made available after the final subission deadline:

Elias Tappiner, Christian Gapp, Martin Welk and Rainer Schuber (2024) CNNs to the Rescue: Head and Neck Tumor Segmentation on MRIs with staged nnU-Nets


The repository basically follows the official [nnU-Net repo](https://github.com/MIC-DKFZ/nnUNet/tree/v2.4.2) with a few adaptation to guide the patch sampling based on an additional input channel.

Start with setting environmentvariables and creating the dataset folder as described in the installation instruction of the nnU-Net repository.

nnU-Net configuration and training for the pre-RT task can be run as follows (assuming the hntsmrg dataset id is 030):

- `nnUNetv2_extract_fingerprint -d 30 --verify_dataset_integrity`
- `nnUNetv2_plan_experiment -d 30 -pl nnUNetPlannerResEncM -gpu_memory_target 24 -overwrite_plans_name nnUNetResEncUNetMPlans_24Gb`
- `nnUNetv2_preprocess -d 30 -plans_name nnUNetResEncUNetMPlans_24Gb -c 3d_fullres -np 12 --verbose`
- `CUDA_VISIBLE_DEVICES=X nnUNetv2_train 30 3d_fullres [0,1,2,3,4,all] -p nnUNetResEncUNetMPlans_24Gb`

for the mid-RT we relay on our changes to train a second-stage network using the data of the pre-RT as additional input. Dataset of the mid-RT task requires three input modalidies:
0000 -> mid-RT MRI
0001 -> registred pre-RT MIR
0002 -> registred pre-RT reference (modality which is used to guide the training process)

After setting up the dataset the nnU-Net configuration and training for the mid-RT task can be run as follows (assuming the hntsmrg dataset id is 031):

... 




When in doubt just reach out, we are happy to help (the repo is just set up as a quick and dirty reference to show how wo got to our results):-) 
