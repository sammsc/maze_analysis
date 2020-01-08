#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# SCRIPT INFORMATION:
# ======================================================================
# SCRIPT: FIRST LEVEL GLM
# PROJECT: MAZE
# ======================================================================
# IMPORT RELEVANT PACKAGES
# ======================================================================
# import basic libraries:
import os
import sys
#import yaml
#import logging
import warnings
from os.path import join as opj
# import nipype libraries:
from nipype.interfaces.utility import Function, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.utils.profiler import log_nodes_cb
from nipype import config, logging
# import spm and matlab interfaces:
from nipype.algorithms.modelgen import SpecifySPMModel, SpecifyModel
from nipype.interfaces.spm.model import (
     Level1Design, EstimateModel, EstimateContrast, ThresholdStatistics,
     Threshold)
from nipype.interfaces.spm import (Level1Design, EstimateModel, EstimateContrast, Threshold)
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces import spm
# import fsl interfaces:
from nipype.workflows.fmri.fsl import create_susan_smooth
from nipype.interfaces.fsl.utils import ExtractROI
# from nipype.interfaces.fsl import model
from  nipype.interfaces import fsl, ants
from nipype.interfaces.base import Bunch
# import libraries for bids interaction:
from bids.layout import BIDSLayout
# import custom functions:
#from highspeed_glm_functions import get_subject_info, plot_stat_maps
import pandas as pd
from pathlib import Path
import nipype.algorithms.modelgen as model   # model generation
from  nipype.interfaces import fsl, ants
import os,json,glob,sys
import numpy
import nibabel
#import nilearn.plotting



def get_subject_info(subject_id, events, confounds, run):

    """
    FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION

    :param events: list with paths to events files
    :param confounds: list with paths to confounds files
    :param run: the task run
    :return: Bunch object with event onsets, durations and regressors
    """

    import pandas as pd
    from nipype.interfaces.base import Bunch

    # sort the inputs chronologically:
    confounds_sorted = sorted(confounds)

    # event types we consider:
    event_names = ['Dummy', 'Tapping']

    # read the events and confounds files of the current run:
    run_confounds = pd.read_csv(confounds_sorted[run], sep="\t")
    
    event_onsets = [[0], events[(events.subject==subject_id) & (events.run==run)].time.tolist()]
    durations = [[8], [0]]

    # define confounds to include as regressors:
    confounds = ['trans', 'rot', 'a_comp_cor', 'cosine']

    # search for confounds of interest in the confounds data frame:
    regressor_names = [col for col in run_confounds.columns if
                       any([conf in col for conf in confounds])]

    # create a nested list with regressor values
    regressors = [list(run_confounds[conf]) for conf in regressor_names]

    # create a bunch for each run:
    run_bunch = Bunch(
        conditions=event_names, onsets=event_onsets, durations=durations,
        regressor_names=regressor_names, regressors=regressors)

    return run_bunch


def get_events():
    action_btn = pd.read_pickle(Path(path_sys / 'img_analysis/nipy/action_btn.pkl'))

    # confounds = pd.read_csv(os.path.join(datadir, "derivatives", "fmriprep",
    #                                      "sub-%s" % source_epi.subject, "ses-%s" % source_epi.session, "func",
    #                                      "sub-%s_ses-%s_task-fingerfootlips_bold_confounds.tsv" % (source_epi.subject,
    #                                                                                                source_epi.session)),
    #                         sep="\t", na_values="n/a")
    #
    # subjectinfo = [Bunch(conditions=condition_names,
    #                      onsets=[onset1],
    #                      durations=[[0]],
    #                      amplitudes=None,
    #                      tmod=None,
    #                      pmod=None,
    #                      regressors=[list(confounds.FramewiseDisplacement.fillna(0)[4:]),
    #                                  list(confounds.aCompCor0[4:]),
    #                                  list(confounds.aCompCor1[4:]),
    #                                  list(confounds.aCompCor2[4:]),
    #                                  list(confounds.aCompCor3[4:]),
    #                                  list(confounds.aCompCor4[4:]),
    #                                  list(confounds.aCompCor5[4:])
    #                                  ],
    #                      regressor_names=['FramewiseDisplacement',
    #                                       'aCompCor0',
    #                                       'aCompCor1',
    #                                       'aCompCor2',
    #                                       'aCompCor3',
    #                                       'aCompCor4',
    #                                       'aCompCor5'])]
    return action_btn




# ======================================================================
# ENVIRONMENT SETTINGS (DEALING WITH ERRORS AND WARNINGS):
# ======================================================================
# set the fsl output type environment variable:
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
# deal with nipype-related warnings:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# filter out warnings related to the numpy package:
warnings.filterwarnings("ignore", message="numpy.dtype size changed*")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed*")


# ======================================================================
# LOAD EXECUTION PARAMETERS FROM THE YAML-FILE:
# ======================================================================
# set paths to the yaml file depending on the operating system:
#if 'darwin' in sys.platform:
#    path_yaml = '/Users/wittkuhn/highspeed/highspeed_analysis/scripts/glm/highspeed_glm_paramaters.yaml'
#elif 'linux' in sys.platform:
#    path_yaml = opj('/home', 'mpib', 'wittkuhn', 'highspeed', 'code', 'highspeed_glm_paramaters.yaml')
## load the yaml files with parameters
#with open('parameters.yaml', 'rb') as f:
#    params = yaml.load(f)


# ======================================================================
# SET PATHS FOR SPM AND MATLAB:
# ======================================================================
# set paths to matlab and spm depending on the operating system:
if 'win32' in sys.platform:
    path_spm = '/Users/Shared/spm12'
    path_matlab = '/Applications/MATLAB_R2017a.app/bin/matlab -nodesktop -nosplash'
elif 'linux' in sys.platform:
    path_spm = opj('/home', 'mpib', 'wittkuhn', 'tools', 'matlab', 'spm12')
    path_matlab = '/opt/matlab/R2017b/bin/matlab -nodesktop -nosplash'
# set paths for spm:
MatlabCommand.set_default_paths(path_spm)
MatlabCommand.set_default_matlab_cmd(path_matlab)
spm.SPMCommand.set_mlab_paths(paths= path_spm, matlab_cmd = path_matlab)


# ======================================================================
# DEFINE PBS CLUSTER JOB TEMPLATE (NEEDED WHEN RUNNING ON THE CLUSTER):
# ======================================================================
job_template = """
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o $HOME/logs/glm
#PBS -m n
#PBS -v FSLOUTPUTTYPE=NIFTI_GZ
source /etc/bash_completion.d/virtualenvwrapper
workon highspeed
module load fsl
module load matlab/R2017b
"""


# ======================================================================
# DEFINE PATHS AND SUBJECTS
# ======================================================================
# define paths depending on the operating system (OS) platform:
if 'win32' in sys.platform:
    path_sys = Path('K:/Users/sam')
    path_img = Path(path_sys / 'maze_state/derivatives/preprocessing/smooth_mni_6mm')
    path_root = str(path_sys)
    sub_list = ['sub-1101']
elif 'linux' in sys.platform:
    path_root = opj('/home', 'beegfs', 'chien', 'maze')
    # grab the list of subjects from the bids data set:
    layout = BIDSLayout(opj(path_root, 'bidsdata'))
    # get all subject ids:
    sub_list = sorted(layout.get_subjects())
    # create a template to add the "sub-" prefix to the ids
    sub_template = ['sub-'] * len(sub_list)
    # add the prefix to all ids:
    sub_list = ["%s%s" % t for t in zip(sub_template,sub_list)]
    # if user defined to run specific subject
    subject_list = [sub_list[int(sys.argv[1])]] if len(sys.argv) >= 2 else sub_list
    sub_list = [('sub-%s' % sys.argv[1])]
else:
    raise Exception('wrong system')
    
    
# ======================================================================
# SETTING UP LOGGING
# ======================================================================
path_log = opj(path_root, 'logs', 'l1analyis')
# enable debug mode for logging and configuration:
config.enable_debug_mode()
# enable logging to file and provide path to the logging file:
config.update_config({'logging': {'log_directory': path_log,
                                  'log_to_file': True},
                      'execution': {'stop_on_first_crash': False,
                                    'keep_unnecessary_outputs': 'false'},
                      'monitoring': {'enabled': True}
                      })
# update the global logging settings:
logging.update_logging(config)


# ======================================================================
# DEFINE SETTINGS
# ======================================================================
# time of repetition, in seconds:
TR = 2
# total number of runs:
num_runs = 10
# smoothing kernel, in mm:
fwhm = 6
# number of dummy variables to remove from each run:
num_dummy = 0
# level 1 contrasts:
stimulus = ('stimulus', 'T', ['stimulus'], [1])
l1contrasts_list = [stimulus]


# ======================================================================
# DEFINE NODE: INFOSOURCE
# ======================================================================
# define the infosource node that collects the data:
infosource = Node(IdentityInterface(
    fields=['subject_id']), name='infosource')
# let the node iterate (paralellize) over all subjects:
infosource.iterables = [('subject_id', sub_list)]


# ======================================================================
# DEFINE SELECTFILES NODE
# ======================================================================
# define all relevant files paths:
templates = dict(
    confounds=opj(path_root, 'derivatives', 'fmriprep', '{subject_id}',
                  'func', '*maze*confounds_regressors.tsv'),
    # events=opj(path_root, 'new_bids', '{subject_id}', '*', 'func',
    #            '*events.tsv'),
    func=opj(path_root, 'derivatives', 'fmriprep', '{subject_id}',
             'func', '*maze*space-T1w*preproc_bold.nii.gz'),
    anat=opj(path_root, 'derivatives', 'fmriprep', '{subject_id}',
             'anat', '{subject_id}_desc-preproc_T1w.nii.gz'),
    wholemask=opj(path_root, 'derivatives', 'fmriprep', '{subject_id}',
                  'func', '*maze*space-T1w*brain_mask.nii.gz')
)
# define the selectfiles node:
selectfiles = Node(SelectFiles(templates, sort_filelist=True),
                   name='selectfiles')
# set expected thread and memory usage for the node:
selectfiles.interface.num_threads = 1
selectfiles.interface.estimated_memory_gb = 0.1


# ======================================================================
# DEFINE NODE: FUNCTION TO GET ACTION EVENT TIMING
# ======================================================================
events_info = Node(Function(
    input_names=None,
    output_names=['events_info'],
    function=get_events),
    name='events_info')


# ======================================================================
# DEFINE NODE: FUNCTION TO GET THE SUBJECT-SPECIFIC INFORMATION
# ======================================================================
subject_info = MapNode(Function(
    input_names=['subject_id', 'events', 'confounds', 'run'],
    output_names=['subject_info'],
    function=get_subject_info),
    name='subject_info', iterfield = ['run'])
subject_info.inputs.run = range(num_runs)
# set expected thread and memory usage for the node:
subject_info.interface.num_threads = 1
subject_info.interface.estimated_memory_gb = 0.1


# ======================================================================
# DEFINE NODE: REMOVE DUMMY VARIABLES (USING FSL ROI)
# ======================================================================
# function: extract region of interest (ROI) from an image
trim = MapNode(ExtractROI(), name='trim', iterfield=['in_file'])
# define index of the first selected volume (i.e., minimum index):
trim.inputs.t_min = num_dummy
# define the number of volumes selected starting at the minimum index:
trim.inputs.t_size = -1
# define the fsl output type:
trim.inputs.output_type = 'NIFTI'
# set expected thread and memory usage for the node:
trim.interface.num_threads = 1
trim.interface.estimated_memory_gb = 3


# ======================================================================
# DEFINE NODE: SPECIFY SPM MODEL (GENERATE SPM-SPECIFIC MODEL)
# ======================================================================
# function: makes a model specification compatible with spm designers
# adds SPM specific options to SpecifyModel

# SpecifyModel - Generates SPM-specific Model
modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                 input_units='secs',
                                 output_units='secs',
                                 time_repetition=TR,
                                 high_pass_filter_cutoff=128),
                 name="modelspec")

l1model = Node(model.SpecifyModel(input_units='secs',
                                 time_repetition=TR,
                                 high_pass_filter_cutoff=128), name="l1model")
# input: concatenate runs to a single session (boolean, default: False):
# l1model.inputs.concatenate_runs = False
# input: units of event onsets and durations (secs or scans):
# l1model.inputs.input_units = 'secs'
# input: units of design event onsets and durations (secs or scans):
# l1model.inputs.output_units = 'secs'
# input: time of repetition (a float):
# l1model.inputs.time_repetition = TR
# high-pass filter cutoff in secs (a float, default = 128 secs):
# l1model.inputs.high_pass_filter_cutoff = 128


# ======================================================================
# DEFINE NODE: LEVEL 1 DESIGN (GENERATE AN SPM DESIGN MATRIX)
# ======================================================================
# function: generate an SPM design matrix

# Level1Design - Generates an SPM design matrix
level1design = Node(Level1Design(bases={'hrf': {'derivs': [0, 0]}},
                                 timing_units='secs',
                                 interscan_interval=TR,
                                 model_serial_correlations='AR(1)'),
                    name="level1design")

l1design = Node(Level1Design(), name="l1design")
# input: (a dictionary with keys which are 'hrf' or 'fourier' or
# 'fourier_han' or 'gamma' or 'fir' and with values which are any value)
l1design.inputs.bases = {'hrf': {'derivs': [0, 0]}}
# input: units for specification of onsets ('secs' or 'scans'):
l1design.inputs.timing_units = 'secs'
# input: interscan interval / repetition time in secs (a float):
l1design.inputs.interscan_interval = TR
# input: Model serial correlations AR(1), FAST or none:
l1design.inputs.model_serial_correlations = 'AR(1)'
# input: number of time-bins per scan in secs (an integer):
l1design.inputs.microtime_resolution = 16
# input: the onset/time-bin in seconds for alignment (a float):
l1design.inputs.microtime_onset = 1
# set expected thread and memory usage for the node:
l1design.interface.num_threads = 1
l1design.interface.estimated_memory_gb = 2


# ======================================================================
# DEFINE NODE: ESTIMATE MODEL (ESTIMATE THE PARAMETERS OF THE MODEL)
# ======================================================================
# function: use spm_spm to estimate the parameters of a model
# EstimateModel - estimate the parameters of the model
level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                      name="level1estimate")

l1estimate = Node(EstimateModel(), name="l1estimate")
# input: (a dictionary with keys which are 'Classical' or 'Bayesian2'
# or 'Bayesian' and with values which are any value)
l1estimate.inputs.estimation_method = {'Classical': 1}
# set expected thread and memory usage for the node:
l1estimate.interface.num_threads = 1
l1estimate.interface.estimated_memory_gb = 2


# ======================================================================
# DEFINE NODE: ESTIMATE CONTRASTS (ESTIMATES THE CONTRASTS)
# ======================================================================
# function: use spm_contrasts to estimate contrasts of interest
l1contrasts = Node(EstimateContrast(), name="l1contrasts")
# input: list of contrasts with each contrast being a list of the form:
# [('name', 'stat', [condition list], [weight list], [session list])]:
l1contrasts.inputs.contrasts = l1contrasts_list
# node input: overwrite previous results:
l1contrasts.overwrite = True
# set expected thread and memory usage for the node:
l1contrasts.interface.num_threads = 1
l1contrasts.interface.estimated_memory_gb = 1.5


# ======================================================================
# DEFINE NODE: FUNCTION TO PLOT CONTRASTS
# ======================================================================
#plot_contrasts = MapNode(Function(
#    input_names=['anat', 'stat_map', 'thresh'],
#    output_names=['out_path'],
#    function=plot_stat_maps),
#    name='plot_contrasts', iterfield = ['thresh'])
## input: plot data with set of different thresholds:
#plot_contrasts.inputs.thresh = [None, 1, 2, 3]
## set expected thread and memory usage for the node:
#plot_contrasts.interface.num_threads = 1
#plot_contrasts.interface.estimated_memory_gb = 0.2


# ======================================================================
# DEFINE NODE: THRESHOLD
# ======================================================================
# function: Topological FDR thresholding based on cluster extent/size.
# Smoothness is estimated from GLM residuals but is assumed to be the
# same for all of the voxels.
thresh = Node(Threshold(), name="thresh")
# input: whether to use FWE (Bonferroni) correction for initial threshold
# (a boolean, nipype default value: True):
thresh.inputs.use_fwe_correction = True
# input: whether to use FDR over cluster extent probabilities (boolean)
thresh.inputs.use_topo_fdr = True
 # input: value for initial thresholding (defining clusters):
thresh.inputs.height_threshold = 0.05
# input: is the cluster forming threshold a stat value or p-value?
# ('p-value' or 'stat', nipype default value: p-value):
thresh.inputs.height_threshold_type = 'p-value'
# input: which contrast in the SPM.mat to use (an integer):
thresh.inputs.contrast_index = 1
# input: p threshold on FDR corrected cluster size probabilities (float):
thresh.inputs.extent_fdr_p_threshold = 0.05
# input: minimum cluster size in voxels (an integer, default = 0):
thresh.inputs.extent_threshold = 0
# set expected thread and memory usage for the node:
thresh.interface.num_threads = 1
thresh.interface.estimated_memory_gb = 0.2


# ======================================================================
# DEFINE NODE: THRESHOLD STATISTICS
# ======================================================================
# function: Given height and cluster size threshold calculate
# theoretical probabilities concerning false positives
thresh_stat = Node(ThresholdStatistics(), name="thresh_stat")
# input: which contrast in the SPM.mat to use (an integer):
thresh_stat.inputs.contrast_index = 1


# ======================================================================
# CREATE DATASINK NODE (OUTPUT STREAM):
# ======================================================================
# create a node of the function:
l1datasink = Node(DataSink(), name='datasink')
# assign the path to the base directory:
l1datasink.inputs.base_directory = opj(path_root, 'derivatives', 'l1pipeline')
# create a list of substitutions to adjust the file paths of datasink:
substitutions = [('_subject_id_', '')]
# assign the substitutions to the datasink command:
l1datasink.inputs.substitutions = substitutions
# determine whether to store output in parameterized form:
l1datasink.inputs.parameterization = True
# set expected thread and memory usage for the node:
l1datasink.interface.num_threads = 1
l1datasink.interface.estimated_memory_gb = 0.2


# ======================================================================
# DEFINE THE LEVEL 1 ANALYSIS SUB-WORKFLOW AND CONNECT THE NODES:
# ======================================================================
# initiation of the 1st-level analysis workflow:
l1analysis = Workflow(name='l1analysis')
# connect the 1st-level analysis components
l1analysis.connect(l1model, 'session_info', l1design, 'session_info')
l1analysis.connect(l1design, 'spm_mat_file', l1estimate, 'spm_mat_file')
l1analysis.connect(l1estimate, 'spm_mat_file', l1contrasts, 'spm_mat_file')
l1analysis.connect(l1estimate, 'beta_images', l1contrasts, 'beta_images')
l1analysis.connect(l1estimate, 'residual_image', l1contrasts, 'residual_image')


# ======================================================================
# DEFINE META-WORKFLOW PIPELINE:
# ======================================================================
# initiation of the 1st-level analysis workflow:
l1pipeline = Workflow(name='l1pipeline')
# stop execution of the workflow if an error is encountered:
l1pipeline.config = {'execution': {'stop_on_first_crash': True}}
# define the base directory of the workflow:
l1pipeline.base_dir = opj(path_root, 'work')
# connect infosource to selectfiles node:
l1pipeline.connect(infosource, 'subject_id', selectfiles, 'subject_id')
# generate subject specific events and regressors to subject_info:
l1pipeline.connect(events_info, 'events_info', subject_info, 'events')
l1pipeline.connect(selectfiles, 'confounds', subject_info, 'confounds')
# connect functional files to smoothing workflow:
# l1pipeline.connect(selectfiles, 'func', susan, 'inputnode.in_files')
# l1pipeline.connect(selectfiles, 'wholemask', susan, 'inputnode.mask_file')
# l1pipeline.connect(susan, 'outputnode.smoothed_files', l1datasink, 'smooth')
# # connect smoothed functional data to the trimming node:
# l1pipeline.connect(susan, 'outputnode.smoothed_files', trim, 'in_file')


# ======================================================================
# INPUT AND OUTPUT STREAM FOR THE LEVEL 1 SPM ANALYSIS SUB-WORKFLOW:
# ======================================================================
# connect regressors to the level 1 model specification node:
l1pipeline.connect(subject_info, 'subject_info', l1analysis, 'l1model.subject_info')
# connect smoothed and trimmed data to the level 1 model specification:
l1pipeline.connect(trim, 'roi_file', l1analysis, 'l1model.functional_runs')
# connect the anatomical image to the plotting node:
#l1pipeline.connect(selectfiles, 'anat', plot_contrasts, 'anat')
# connect spm t-images to the plotting node:
#l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', plot_contrasts, 'stat_map')
# connect the t-images and spm mat file to the threshold node:
l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', thresh, 'stat_image')
l1pipeline.connect(l1analysis, 'l1contrasts.spm_mat_file', thresh, 'spm_mat_file')
# connect all output results of the level 1 analysis to the datasink:
l1pipeline.connect(l1analysis, 'l1estimate.beta_images', l1datasink, 'estimates.@beta_images')
l1pipeline.connect(l1analysis, 'l1estimate.residual_image', l1datasink, 'estimates.@residual_image')
l1pipeline.connect(l1analysis, 'l1contrasts.spm_mat_file', l1datasink, 'contrasts.@spm_mat')
l1pipeline.connect(l1analysis, 'l1contrasts.spmT_images', l1datasink, 'contrasts.@spmT')
l1pipeline.connect(l1analysis, 'l1contrasts.con_images', l1datasink, 'contrasts.@con')
#l1pipeline.connect(plot_contrasts, 'out_path', l1datasink, 'contrasts.@out_path')
l1pipeline.connect(thresh, 'thresholded_map', l1datasink, 'thresh.@threshhold_map')
l1pipeline.connect(thresh, 'pre_topo_fdr_map', l1datasink, 'thresh.@pre_topo_fdr_map')


# ======================================================================
# WRITE GRAPH AND EXECUTE THE WORKFLOW
# ======================================================================
# write the graph:
l1pipeline.write_graph(graph2use='colored', format='png', simple_form=True)
# set the maximum resources the workflow can utilize:
#args_dict = {'status_callback' : log_nodes_cb}
# execute the workflow depending on the operating system:
# if 'darwin' in sys.platform:
#     # will execute the workflow using all available cpus:
#     l1pipeline.run(plugin='MultiProc')
# elif 'linux' in sys.platform:
#     l1pipeline.run(plugin='PBS', plugin_args=dict(template=job_template))
