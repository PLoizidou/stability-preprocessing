# stability-preprocessing

preprocessing code for one-photon calcium imaging tracking the stability of hippocampal activity

## Setup
To get started, run 
```conda env create -f environment.yml && conda activate stability-preprocessing```
to install and activate the conda environment.

## Components
[CaImAn](https://caiman.readthedocs.io/en/latest/ "CaImAn docs") is used for calcium data extraction via the CNMF-E algorithm.

[DeepLabCut](https://deeplabcut.github.io/DeepLabCut/README.html "DeepLabCut docs") is used for pose estimation.

The [Neurodata Without Borders](https://pynwb.readthedocs.io/en/stable/ "PyNWB docs") standard is used for data storage. Videos are stored externally to the `.nwb` file in their original format and linked in their respetive collections' in the' external data fields.

## Usage

Workflow to process a new session follows the following steps:

**NOTE:** Be *very* careful to clear the outputs of jupyter notebooks before adding the notebooks to a git commit. Leaving the outputs in will cause storage problems for your repo because the images are too big.

### DLC training
This pipeline assumes you've already trained a DeepLabCut model for your videos.

### CaImAn parameter setting
You probably only need to run this the first time you're collecting data for a subject, if you notce motion correction artifacts, or if you change camera recording parameters (e.g., gain). `scripts/demo_pipeline.ipynb` will help you select your CNMF-E parameters, and `scripts/demo_motion_correction.ipynb` will help you select motion correction parameters. Both can be run on a raw miniscope `.avi` file. Ideal parameters may vary between subjects, so make sure to record each subject's parameter configuration.

### Process
Once you've selected parameters for your subject, you can preprocess your data using `scripts/preproc_caiman.py`. Assuming you're running it from the home directory, you can run ```python scripts/preproc_caiman.py --input_path <your-avi-file> --output_path <your-session-output-file>```. You can view additional parameters by running ```python scripts/preproc_caiman.py --help```. This will preprocess one `.avi` file and populate the output directory with a `caiman_results.hdf5` file containing components, their calcium outputs, and intermediate files like memmapped motion correction outputs.

### Visualize results
When you'd like to visualize the components and their traces, run the `scripts/visualize_caiman_results.ipynb` notebook. Once you set the results path, select run all. The notebook shows a subset of components because of memory constraints.