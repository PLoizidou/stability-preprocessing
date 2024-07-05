# stability-preprocessing

preprocessing code for one-photon calcium imaging tracking the stability of hippocampal activity

## Setup
To get started, run 
```conda env create -f environment.yml && conda activate stability-preprocessing```
to install and activate the conda environment.

## Components
[Caiman](https://caiman.readthedocs.io/en/latest/ "Caiman docs") is used for calcium data extraction via the CNMF-E algorithm.

[DeepLabCut](https://deeplabcut.github.io/DeepLabCut/README.html "DeepLabCut docs") is used for pose estimation.

The [Neurodata Without Borders](https://pynwb.readthedocs.io/en/stable/ "PyNWB docs") standard is used for data storage. Videos are stored externally to the `.nwb` file in their original format and linked in their respetive collections' in the' external data fields.

[NeuroConv](https://neuroconv.readthedocs.io/en/main/index.html "NeuroConv docs") is used for conversion of DLC outputs to NWB standard, and is a good place to look for future conversions if necessary.

## Usage

Workflow to process a new session follows the following steps. You must run `conda activate stability-preprocessing` for the correct environment for almost all scripts. The one exception is `extract_pose.py`, which requires `conda activate dlc`.

**NOTE:** Be ***very*** careful to clear the outputs of jupyter notebooks before adding the notebooks to a git commit. Leaving the outputs in will cause storage problems for your repo because the images are too big.

### DLC training
This pipeline assumes you've already trained a DeepLabCut model for your videos.

Training is probably easiest do perform using the DLC GUI because of labelling. Start with a single animal project and default settings for this process. If you run into out-of-memory (OOM) errors during this process, decrease the batch size by editing the config.yml in the GUI. Full guide to single animal projects [here](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html). Hold off on analyzing any videos until you've read the rest of this because the GUI doesn't store the outputs in the NWB format.

### CaImAn parameter setting
You probably only need to run this the first time you're collecting data for a subject, if you notce motion correction artifacts, or if you change camera recording parameters (e.g., gain). `scripts/demo_pipeline.ipynb` will help you select your CNMF-E parameters, and `scripts/demo_motion_correction.ipynb` will help you select motion correction parameters. Both can be run on a raw miniscope `.avi` file. Ideal parameters may vary between subjects, so make sure to record each subject's parameter configuration. 

The best way of doing this would be to incorporate something like [Mesmerize](https://mesmerize-core.readthedocs.io/en/latest/index.html "Mezmerize docs"), which streamlines parameter tuning.

**NOTE:** Using a subject's parameters requires providing them as arguments to downstream scripts or (even worse) explicityly changing bash scripts. The ideal way to manage this situation is by [setting up .yaml configs](https://stackoverflow.com/questions/38404633/reading-yaml-config-file-in-python-and-using-variables) for each subject that contain all the arguments for CaImAn. This will both help record the parameters and ensure there's a single soure of truth for processing parameters.

### Session curation
Data are grouped by mouse when collected, and it will be easier from an organization and processing perspective to group them by session date/time. `scripts/curate_trials.py` will do this, and can be run as:
```
python scripts/curate_nwb.py path/to/raw/data/root path/to/nwb/data/root --start_data <YYYY-MM-DD> --animals <Animal1> <Animal2>
```
This assumes your folder structure is 
```
path/to/raw/data/root/
    Animal1/
        miniscope2024-05-28T13_15_08.avi
        timestamps2024-05-28T13_15_08.csv
        behavior2024-05-28T13_15_08.avi
        miniscope2024-05-29T13_15_08.avi
        timestamps2024-05-29T13_15_08.csv
        behavior2024-05-29T13_15_08.avi
        ...
    Animal2/
        ...
```
after collection and will output

```
path/to/nwb/data/root/
    Animal1/
        ses-20240528T131508/
            miniscope2024-05-28T13_15_08.avi
            timestamps2024-05-28T13_15_08.csv
            behavior2024-05-28T13_15_08.avi
        ses-20240529T131508/
            miniscope2024-05-29T13_15_08.avi
            timestamps2024-05-29T13_15_08.csv
            behavior2024-05-29T13_15_08.avi
        ...
    Animal2/
        ...
```
Which will contain all the same data and can be used for subsequent analyses. Optionally, a `--save_nwb` flag will save an additional `.nwb` file containing references the necessary files and additional trial info.

### Process
**NOTE**: The script `scripts/preproc_session.sh` will run this full pipeline given the path to an NWB file `bash scripts/preproc_session.sh path/to/miniscope path/to/home_video path/to/linear_video`. This also provides an example of how to run each of the scripts individually, but you should read the rest of this section before working with this script. Running it does not require you to activate a conda environment beforehand. This is handled internally.

Once you've selected parameters for your subject, you can preprocess your data using `scripts/preproc_caiman.py`. Assuming you're running it from the home directory, you can run ```python scripts/preproc_caiman.py --input_path <your-avi-file> --output_path <your-session-output-file>```. You can view additional parameters by running ```python scripts/preproc_caiman.py --help```. Outputs will be stored in `caiman/caiman_results.hdf5` within the trial directory.

A pair of scripts exist to run DLC to extract pose and convert those outputs to NWB format. Those can be found in `scripts/estimate_pose.py` and `scripts/curate_pose_nwb.py`, respectively. Raw DLC outputs will be stored in a `/dlc` directory in the same directory as the NWB file after `scripts/estimate_pose.py` and then moved to an NWB file after `scripts/curate_pose_nwb.py`. They are run as:

```
python scripts/estimate_pose.py path/to/dlc/config path/to/my/video.avi --gpu_id 0
``` 
and, *optionally*,
```
python scripts/curate_pose_nwb.py path/to/dlc/config path/to/my/data.nwb behavioral_acquisition_key pose_storage_key --match_regex regex_for_dlc_output
```
where behavioral_acquisition_key and pose_storage_key are the names the data has in the NWB acquisition module, and the name the results will be saved to in the NWB behavior module, respectively. `--gpu_id` is used to make a GPU visibble for DLC and `--match_regex` allows you to narrow the filed in `/dlc` that will be considered conversion and storage. Make sure the regex you pick matches only one file, probably by using some version like `*resnet50_<DLC_PROJECT_NAME>*.h5`.


### Visualize results
When you'd like to visualize the components and their traces, run the `scripts/visualize_caiman_results.ipynb` notebook. Once you set the results path, select run all. The notebook shows a subset of components because of memory constraints.

## Downstream analyses
See `scripts/downstream.ipynb` to see how to extract the data of interest.