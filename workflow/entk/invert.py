#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import nested_scopes, generators, division, absolute_import, \
    with_statement, print_function
import os
import os.path as p
from radical.entk import Pipeline, Stage, Task, AppManager
# import traceback
# import sys
from source import CMTSource
from gcmt3d.workflow.create_database_entry import create_entry
from gcmt3d.workflow.prepare_path_files import make_paths
from get_conversion_list import get_conversion_list
from get_process_dict import get_process_dict
from get_window_list import get_window_list

from read_yaml import read_yaml_file

mainparams = read_yaml_file(p.join(p.dirname(p.abspath(__file__)), "entk.yml"))

radical_dict = mainparams["RADICAL"]

radical_dict["RADICAL_PILOT_DBURL"] = \
    (f"mongodb://{radical_dict['RMQ_USERNAME']}:"
     f"{radical_dict['RMQ_PASSWORD']}@129.114.17.185/specfm")

for var, val in radical_dict.items():
    os.environ[var] = val

hostname = radical_dict["RMQ_HOSTNAME"]
port = int(radical_dict["RMQ_PORT"])
password = radical_dict["RMQ_PASSWORD"]
username = radical_dict["RMQ_USERNAME"]

database_dict = mainparams["DATABASE"]

# Specfem
specfem = database_dict["SPECFEM"]

# GCMT3D
GCMT3D = database_dict["GCMT3D"]
WORKFLOW_DIR = f"{GCMT3D}/workflow"
PARAMETER_PATH = f"{WORKFLOW_DIR}/params"
SLURM_DIR = f"{WORKFLOW_DIR}/slurmwf"
BIN_DIR = f"{WORKFLOW_DIR}/bins"

# DATABASE
DATABASE_DIR = database_dict['DIR']


cmtfile = "/tigress/lsawade/source_inversion_II/events/CMT.perturb.440/C200605061826B"
# EARTHQUAKE PARAMETERS and Paths derived from the file
# Getting database entry from CMTSOLUTION FILE
cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)
CID = cmtsource.eventname
CIN_DB = f"{DATABASE_DIR}/{CID}/{CID}.cmt"
CDIR = p.dirname(CIN_DB)            # Earthquake Directory
CMT_SIM_DIR = f"{CDIR}/CMT_SIMs"      # Simulation directory

# directory
PROCESSED = f"{CDIR}/seismograms/processed_seismograms"
SEISMOGRAMS = f"{CDIR}/seismograms"
STATION_DATA = f"{CDIR}/station_data"
LOG_DIR = f"{CDIR}/logs"  # Logging directory
INVERSION_OUTPUT_DIR = f"{CDIR}/inversion"
WINDOW_PATHS = f"{CDIR}/workflow_files/path_files/window_paths"
PROCESS_PATHS = f"{CDIR}/workflow_files/path_files/process_paths"
CONVERSION_PATHS = f"{CDIR}/workflow_files/path_files/conversion_paths"

SYNTHETIC_FILES = database_dict["SYNTHETIC"]
OBSERVED_DATA = database_dict["OBSERVED"]
OBSERVED_FILES = f"{OBSERVED_DATA}/waveform"
STATION_FILES = f"{OBSERVED_DATA}/station"

ANACONDA = database_dict["ENVIRONMENT"]

# Sorting things (not so important right now)
# INVERSION_DIRECTORY = "/gpfs/alpine/geo111/proj-shared/lsawade/submission_scripts"
# SUBMITTED = f"{INVERSION_DIRECTORY}/submitted_cmts"
# DONE = f"{INVERSION_DIRECTORY}/done_cmts"

# Create Entry
# create_entry(cmtfile, DATABASE_DIR, PARAMETER_PATH, specfem=False)

# Create Paths
# make_paths(CIN_DB, PARAMETER_PATH, conversion=True)

# Create a Pipeline object
pipe = Pipeline()

# -----------------------------------------------------------------------------
# # Create a Stage object
# download = Stage()

# # Create a Task object
# t = Task()
# t.name = 'Download'  # Assign a name to the task (optional, do not use ',' or '_')
# t.pre_exec = [
#     # Load the compilers and shit
#     "module purge",
#     # Load anaconda and set environment
#     "module load anaconda3",
#     f"conda activate {ANACONDA}",
#     "module load openmpi/gcc"
#     ]
# t.executable = 'request-data'   # Assign executable to the task
# t.arguments = ['-f', CIN_DB, '-p', PARAMETER_PATH]  # Assign arguments for the task executable
# t.download_output_data = ['STDOUT', 'STDERR']
# t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
#               'threads_per_process': 1, 'thread_type': 'OpenMP'}
# t.download_output_data = ['STDOUT', 'STDERR']
# # Add Task to the Stage
# download.add_tasks(t)

# # Add Stage to the Pipeline
# pipe.add_stages(download)

# -----------------------------------------------------------------------------
# Create a Task object
# Create a Stage object
copydata = Stage()

t = Task()
t.name = 'Copy-Data'  # Assign a name to the task (optional, do not use ',' or '_')
t.pre_exec = [
    # Load the compilers and shit
    "module purge",
    # Load anaconda and set environment
    "module load anaconda3",
    f"conda activate {ANACONDA}",
    "module load openmpi/gcc"
    ]
t.executable = 'copy-data-to-database'   # Assign executable to the task
t.arguments = ['-f', CIN_DB,
               '-o', OBSERVED_FILES,
               '-sta', STATION_FILES,
               '-s', SYNTHETIC_FILES]  # Assign arguments for the task executable
t.download_output_data = ['STDOUT', 'STDERR']
t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
              'threads_per_process': 1, 'thread_type': 'OpenMP'}
t.download_output_data = ['STDOUT', 'STDERR']
# Add Task to the Stage
copydata.add_tasks(t)

# Add Stage to the Pipeline
pipe.add_stages(copydata)

# -----------------------------------------------------------------------------

conversion_stage = Stage()
conversion_stage.name = f"Conversion"
for pathfile in get_conversion_list(CONVERSION_PATHS):
    convname = p.basename(pathfile).split(".")[0].capitalize().replace("_", "-")
    t = Task()
    t.name = f'{convname}'  # Assign a name to the task (optional, do not use ',' or '_')
    t.pre_exec = [
        # Load the compilers and shit
        "module purge",
        # Load anaconda and set environment
        "module load anaconda3",
        f"conda activate {ANACONDA}",
        "module load openmpi/gcc"
        ]
    t.executable = 'convert2asdf'   # Assign executable to the task
    t.arguments = ['-f', pathfile]  # Assign arguments for the task executable
    t.download_output_data = ['STDOUT', 'STDERR']
    t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
                  'threads_per_process': 1, 'thread_type': 'OpenMP'}
    t.download_output_data = ['STDOUT', 'STDERR']
    # Add Task to the Stage
    conversion_stage.add_tasks(t)

    pipe.add_stages(conversion_stage)

# -----------------------------------------------------------------------------

for _wave, process_list in get_process_dict(PROCESS_PATHS).items():

    process_stage = Stage()
    process_stage.name = f"Process-{_wave}"

    counter = 0
    for pathfile in process_list:
        t = Task()
        t.name = f'{counter:0>10}'  # Assign a name to the task (optional, do not use ',' or '_')
        t.pre_exec = [
            # Load the compilers and shit
            "module purge",
            # Load anaconda and set environment
            "module load anaconda3",
            f"conda activate {ANACONDA}",
            "module load openmpi/gcc"
            ]
        t.executable = 'process-asdf'   # Assign executable to the task
        t.arguments = ['-f', pathfile]  # Assign arguments for the task executable
        t.download_output_data = ['STDOUT', 'STDERR']
        t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
                      'threads_per_process': 1, 'thread_type': 'OpenMP'}
        t.download_output_data = ['STDOUT', 'STDERR']
        # Add Task to the Stage
        process_stage.add_tasks(t)
        counter += 1

    pipe.add_stages(process_stage)


# -----------------------------------------------------------------------------

window_stage = Stage()
window_stage.name = f"Windowing"
for pathfile in get_window_list(WINDOW_PATHS):
    wave = p.basename(pathfile).split(".")[0].capitalize()
    t = Task()
    t.name = f'{wave}'  # Assign a name to the task (optional, do not use ',' or '_')
    t.pre_exec = [
        # Load the compilers and shit
        "module purge",
        # Load anaconda and set environment
        "module load anaconda3",
        f"conda activate {ANACONDA}",
        "module load openmpi/gcc"
        ]
    t.executable = 'select-windows'   # Assign executable to the task
    t.arguments = ['-f', pathfile]  # Assign arguments for the task executable
    t.download_output_data = ['STDOUT', 'STDERR']
    t.cpu_reqs = {'processes': 14, 'process_type': 'MPI',
                  'threads_per_process': 1, 'thread_type': 'OpenMP'}
    t.download_output_data = ['STDOUT', 'STDERR']
    # Add Task to the Stage
    window_stage.add_tasks(t)

    pipe.add_stages(window_stage)

# -----------------------------------------------------------------------------

# # Create a Stage object
cmt3d = Stage()

# Create a Task object
t = Task()
t.name = 'cmt3d'  # Assign a name to the task (optional, do not use ',' or '_')
t.pre_exec = [
    # Load the compilers and shit
    "module purge",
    # Load anaconda and set environment
    "module load anaconda3",
    f"conda activate {ANACONDA}",
    "module load openmpi/gcc"
    ]
t.executable = 'inversion'   # Assign executable to the task
t.arguments = ['-f', CIN_DB]  # Assign arguments for the task executable
t.download_output_data = ['STDOUT', 'STDERR']
t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
              'threads_per_process': 1, 'thread_type': 'OpenMP'}
t.download_output_data = ['STDOUT', 'STDERR']
# Add Task to the Stage
cmt3d.add_tasks(t)

# Add Stage to the Pipeline
pipe.add_stages(cmt3d)

# -----------------------------------------------------------------------------

# Create a Stage object
g3d = Stage()

# Create a Task object
t = Task()
t.name = 'g3d'  # Assign a name to the task (optional, do not use ',' or '_')
t.pre_exec = [
    # Load the compilers and shit
    "module purge",
    # Load anaconda and set environment
    "module load anaconda3",
    f"conda activate {ANACONDA}",
    "module load openmpi/gcc"
    ]
t.executable = 'gridsearch'   # Assign executable to the task
t.arguments = ['-f', CIN_DB]  # Assign arguments for the task executable
t.download_output_data = ['STDOUT', 'STDERR']
t.cpu_reqs = {'processes': 25, 'process_type': 'MPI',
              'threads_per_process': 1, 'thread_type': 'OpenMP'}
t.download_output_data = ['STDOUT', 'STDERR']
# Add Task to the Stage
g3d.add_tasks(t)

# Add Stage to the Pipeline
pipe.add_stages(g3d)

# -----------------------------------------------------------------------------

# Create a Stage object
result = Stage()

# Create a Task object
t = Task()
t.name = 'plotting'  # Assign a name to the task (optional, do not use ',' or '_')
t.pre_exec = [
    # Load the compilers and shit
    "module purge",
    # Load anaconda and set environment
    "module load anaconda3",
    f"conda activate {ANACONDA}",
    "module load openmpi/gcc"
    ]
t.executable = 'plot-event-summary'   # Assign executable to the task
t.arguments = [f"{INVERSION_OUTPUT_DIR}/cmt3d/*stats.json",
               "-g", f"{INVERSION_OUTPUT_DIR}/g3d/*stats.json",
               "-f", f"{INVERSION_OUTPUT_DIR}/full_summary.pdf"]
t.download_output_data = ['STDOUT', 'STDERR']
t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
              'threads_per_process': 1, 'thread_type': 'OpenMP'}
t.download_output_data = ['STDOUT', 'STDERR']
# Add Task to the Stage
result.add_tasks(t)

# Add Stage to the Pipeline
pipe.add_stages(result)


res_dict = {
    # 'resource': 'local.localhost',  
    'resource': 'princeton.traverse',
    'project_id': 'test',
    'schema': 'local',
    'walltime': 45,
    'cpus': 32 * 1,
    # 'gpus': 4 * 1,
}

appman = AppManager(hostname=hostname, port=port, resubmit_failed=False,
                    username=username, password=password)
appman.resource_desc = res_dict
appman.workflow = set([pipe])
appman.run()
