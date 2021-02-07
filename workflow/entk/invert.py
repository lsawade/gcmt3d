#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import nested_scopes, generators, division, absolute_import, \
    with_statement, print_function
import os
from radical.entk import Pipeline, Stage, Task, AppManager
# import traceback
# import sys
from source import CMTSource
from get_conversion_list import get_conversion_list
from get_process_dict import get_process_dict
from get_window_list import get_window_list
from read_yaml import read_yaml_file

SCRIPT_LOCATION = os.path.abspath(__file__)

# ############### EnTK Parameters to be set and read #########################
ENTK_PARAMS = read_yaml_file(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)), "entk.yml"))

RADICAL_DICT = ENTK_PARAMS["RADICAL"]
RADICAL_DICT["RADICAL_PILOT_DBURL"] = \
    (f"mongodb://{RADICAL_DICT['RMQ_USERNAME']}:"
     f"{RADICAL_DICT['RMQ_PASSWORD']}@129.114.17.185/specfm")

for var, val in RADICAL_DICT.items():
    os.environ[var] = val


hostname = RADICAL_DICT["RMQ_HOSTNAME"]
port = int(RADICAL_DICT["RMQ_PORT"])
password = RADICAL_DICT["RMQ_PASSWORD"]
username = RADICAL_DICT["RMQ_USERNAME"]

# ############### GCMT3D Parameters to be set and read #######################
GCMT3D_PARAMS = ENTK_PARAMS["GCMT3D"]
GCMT3D_DIR = GCMT3D_PARAMS["GCMT3D"]
WORKFLOW_DIR = os.path.join(GCMT3D_DIR, 'workflow')
DOWNLOAD_SH = os.path.join(WORKFLOW_DIR, "entk", "ssh-request-data.sh")
PARAMETER_PATH = GCMT3D_PARAMS["PARAMS"]
CONDA_ENV = GCMT3D_PARAMS["ENVIRONMENT"]

# DATABASE
DATABASE_PARAMS = read_yaml_file(os.path.join(
    PARAMETER_PATH, "Database", "DatabaseParameters.yml"))
DATABASE_DIR = DATABASE_PARAMS['entkdatabase']
DATABASE_DISCARD_DIR = DATABASE_PARAMS['entkdatabase_bad']

# SPECFEM
SPECFEM_PARAMS = read_yaml_file(os.path.join(
    PARAMETER_PATH, "SpecfemParams", "SpecfemParams.yml"))


# ################ EVENT INFORMATION #########################################
cmtfile = os.path.join(os.path.dirname(SCRIPT_LOCATION), "sumatra.cmt")
# EARTHQUAKE PARAMETERS and Paths derived from the file
# Getting database entry from CMTSOLUTION FILE
cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)
CID = cmtsource.eventname
CIN_DB = f"{DATABASE_DIR}/{CID}/{CID}.cmt"
CDIR = os.path.dirname(CIN_DB)            # Earthquake Directory
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


# Create a Pipeline object
p = Pipeline()

# Create Entry ####################################################
s = Stage()
s.name = 'CreateEntryStage'

# Create a Task object
t = Task()
t.name = 'CreateEntryTask'

t.pre_exec = [
    f"conda activate {CONDA_ENV}",
]
t.executable = 'create-entry'
t.arguments = ["-f", f"{CIN_DB}",
               "-d", f"{DATABASE_DIR}",
               "-p", f"{PARAMETER_PATH}"]
t.download_output_data = ['STDOUT', 'STDERR']
t.cpu_reqs = {
    'cpu_processes': 1,
    'cpu_process_type': None,
    'cpu_threads': 1,
    'cpu_thread_type': None
}


def create_postfunc():
    with open(f"{CDIR}/STATUS", 'w') as f:
        f.write("CREATED")


s.post_exec = create_postfunc

# Add Task to Stage and Stage to the Pipeline
s.add_tasks(t)
p.add_stages(s)

# Download ########################################################
s = Stage()
s.name = 'DownloadStage'

# Create a Task object
t = Task()
t.name = 'DownloadTask'

# If The compute node does have internet we can simply run the
# download function 'request-data'
print(RADICAL_DICT['COMPUTE_INTERNET'])
if RADICAL_DICT['COMPUTE_INTERNET']:
    t.pre_exec = [
        f"conda activate {CONDA_ENV}",
        "module load openmpi/gcc"
    ]
    t.executable = 'request-data'
    t.arguments = ["-f", f"{CIN_DB}", "-p", f"{PARAMETER_PATH}"]
    t.download_output_data = ['STDOUT', 'STDERR']
    t.cpu_reqs = {
        'cpu_processes': 1,
        'cpu_process_type': None,
        'cpu_threads': 1,
        'cpu_thread_type': None
    }
# else we have to setup a workaround script to login into the login node
# and download from there
else:
    t.executable = '/usr/bin/ssh'
    t.arguments = ["lsawade@traverse.princeton.edu",
                   f"'bash -l {DOWNLOAD_SH} {CIN_DB} {PARAMETER_PATH}'"]
    t.download_output_data = ['STDOUT', 'STDERR']
    t.cpu_reqs = {
        'cpu_processes': 1,
        'cpu_process_type': None,
        'cpu_threads': 1,
        'cpu_thread_type': None
    }


def download_postfunc():
    with open(f"{CDIR}/STATUS", 'w') as f:
        f.write("DATA_DOWNLOADED")


s.post_exec = download_postfunc

# Add Task to Stage and Stage to the Pipeline
s.add_tasks(t)
p.add_stages(s)

# Conversion ########################################################

s = Stage()
s.name = "Conversion"
for pathfile in get_conversion_list(CONVERSION_PATHS):
    convname = p.basename(pathfile).split(
        ".")[0].capitalize().replace("_", "-")
    t = Task()
    # Assign a name to the task (optional, do not use ',' or '_')
    t.name = f'{convname}'
    t.pre_exec = [
        f"conda activate {CONDA_ENV}",
        "module load openmpi/gcc"
    ]
    t.executable = 'convert2asdf'   # Assign executable to the task
    t.arguments = ['-f', pathfile]  # Assign arguments for the task executable
    t.download_output_data = ['STDOUT', 'STDERR']
    t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
                  'threads_per_process': 1, 'thread_type': 'OpenMP'}
    t.download_output_data = ['STDOUT', 'STDERR']
    # Add Task to the Stage
    s.add_tasks(t)


def conversion_postfunc():
    with open(f"{CDIR}/STATUS", 'w') as f:
        f.write("DATA_CONVERTED")


s.post_exec = conversion_postfunc

p.add_stages(s)


# Simulation ########################################################


CMT_LIST = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp", "CMT_tp",
            "CMT_depth", "CMT_lat", "CMT_lon"]

s = Stage()
s.name = 'SimulationStage'

for _cmt in CMT_LIST:

    # Create Task
    t = Task()
    t.name = f"SIMULATION.{_cmt}"
    tdir = os.path.join(CMT_SIM_DIR, _cmt)
    t.pre_exec = [
        # Load necessary modules
        'module load openmpi/gcc',
        'module load cudatoolkit/11.0',

        # Change to your specfem run directory
        # f'rm -rf {tdir}',
        # f'mkdir {tdir}',
        f'cd {tdir}',

        # # Create data structure in place
        # f'ln -s {SPECFEM_PARAMS["SPECFEM_DIR"]}/bin .',
        # f'ln -s {SPECFEM_PARAMS["SPECFEM_DIR"]}/DATABASES_MPI .',
        # f'cp -r {SPECFEM_PARAMS["SPECFEM_DIR"]}/OUTPUT_FILES .',
        # 'mkdir DATA',
        # f'cp {SPECFEM_PARAMS["SPECFEM_DIR"]}/DATA/CMTSOLUTION ./DATA/',
        # f'cp {SPECFEM_PARAMS["SPECFEM_DIR"]}/DATA/STATIONS ./DATA/',
        # f'cp {SPECFEM_PARAMS["SPECFEM_DIR"]}/DATA/Par_file ./DATA/'
    ]
    t.executable = './bin/xspecfem3D'
    t.cpu_reqs = {
        'cpu_processes': SPECFEM_PARAMS["tasks"], 'cpu_process_type': 'MPI',
        'cpu_threads': 1, 'cpu_thread_type': 'OpenMP'}
    t.gpu_reqs = {
        'gpu_processes': 1, 'gpu_process_type': 'MPI',
        'gpu_threads': 1, 'gpu_thread_type': 'CUDA'}
    t.download_output_data = ['STDOUT', 'STDERR']

    # Add task to stage
    s.add_tasks(t)


def simulation_postfunc():
    with open(f"{CDIR}/STATUS", 'w') as f:
        f.write(f"{_wave.upper()}_PROCESSED")


s.post_exec = simulation_postfunc

p.add_stages(s)


# Processing ########################################################

for _wave, process_list in get_process_dict(PROCESS_PATHS).items():

    s = Stage()
    s.name = f"Process{_wave.capitalize()}"

    counter = 0
    for pathfile in process_list:
        t = Task()
        # Assign a name to the task (optional, do not use ',' or '_')
        t.name = f'{counter:0>10}'
        t.pre_exec = [
            f"conda activate {CONDA_ENV}",
            "module load openmpi/gcc"
        ]
        t.executable = 'process-asdf'   # Assign executable to the task
        # Assign arguments for the task executable
        t.arguments = ['-f', pathfile]
        t.download_output_data = ['STDOUT', 'STDERR']
        t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
                      'threads_per_process': 1, 'thread_type': 'OpenMP'}
        t.download_output_data = ['STDOUT', 'STDERR']
        # Add Task to the Stage
        s.add_tasks(t)
        counter += 1

    def process_postfunc():
        with open(f"{CDIR}/STATUS", 'w') as f:
            f.write(f"{_wave.upper()}_PROCESSED")

    s.post_exec = process_postfunc
    p.add_stages(s)


# Windowing ########################################################

s = Stage()
s.name = "Windowing"
for pathfile in get_window_list(WINDOW_PATHS):
    wave = p.basename(pathfile).split(".")[0].capitalize()
    t = Task()
    # Assign a name to the task (optional, do not use ',' or '_')
    t.name = f'{wave}'
    t.pre_exec = [
        # Load the compilers and shit
        "module purge",
        # Load anaconda and set environment
        "module load anaconda3",
        f"conda activate {CONDA_ENV}",
        "module load openmpi/gcc"
    ]
    t.executable = 'select-windows'   # Assign executable to the task
    t.arguments = ['-f', pathfile]  # Assign arguments for the task executable
    t.download_output_data = ['STDOUT', 'STDERR']
    t.cpu_reqs = {'processes': 14, 'process_type': 'MPI',
                  'threads_per_process': 1, 'thread_type': 'OpenMP'}
    t.download_output_data = ['STDOUT', 'STDERR']
    # Add Task to the Stage
    s.add_tasks(t)


def win_postfunc():
    with open(f"{CDIR}/STATUS", 'w') as f:
        f.write(f"{_wave.upper()}_WINDOWED")


s.post_exec = win_postfunc

p.add_stages(s)

# Inversion ########################################################

# # Create a Stage object
s = Stage()

# Create a Task object
t = Task()
t.name = 'cmt3d'  # Assign a name to the task (optional, do not use ',' or '_')
t.pre_exec = [
    f"conda activate {CONDA_ENV}",
    "module load openmpi/gcc"
]
t.executable = 'inversion'   # Assign executable to the task
t.arguments = ['-f', CIN_DB]  # Assign arguments for the task executable
t.download_output_data = ['STDOUT', 'STDERR']
t.cpu_reqs = {'processes': 1, 'process_type': 'MPI',
              'threads_per_process': 1, 'thread_type': 'OpenMP'}
t.download_output_data = ['STDOUT', 'STDERR']
# Add Task to the Stage
s.add_tasks(t)


def cmt3d_postfunc():
    with open(f"{CDIR}/STATUS", 'w') as f:
        f.write("INVERTED")


s.post_exec = cmt3d_postfunc

# Add Stage to the Pipeline
p.add_stages(s)

# GRIDSEARCH ########################################################

# Create a Stage object
s = Stage()

# Create a Task object
t = Task()
t.name = 'g3d'  # Assign a name to the task (optional, do not use ',' or '_')
t.pre_exec = [
    f"conda activate {CONDA_ENV}",
    "module load openmpi/gcc"
]
t.executable = 'gridsearch'   # Assign executable to the task
t.arguments = ['-f', CIN_DB]  # Assign arguments for the task executable
t.download_output_data = ['STDOUT', 'STDERR']
t.cpu_reqs = {'cpu_processes': 25, 'cpu_process_type': 'MPI',
              'cpu_threads': 1, 'cpu_thread_type': 'OpenMP'}
t.download_output_data = ['STDOUT', 'STDERR']
# Add Task to the Stage
s.add_tasks(t)


def g3d_postfunc():
    with open(f"{CDIR}/STATUS", 'w') as f:
        f.write("GRIDSEARCHED")


s.post_exec = g3d_postfunc

# Add Stage to the Pipeline
p.add_stages(s)

# Results ########################################################

# Create a Stage object
s = Stage()

# Create a Task object
t = Task()
# Assign a name to the task (optional, do not use ',' or '_')
t.name = 'plotting'
t.pre_exec = [
    f"conda activate {CONDA_ENV}",
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
s.add_tasks(t)


def result_postfunc():
    with open(f"{CDIR}/STATUS", 'w') as f:
        f.write("FINISHED")


s.post_exec = g3d_postfunc

# Add Stage to the Pipeline
p.add_stages(s)


# -----------------------------------------------------------------------------

# Create a check that looks att changes between solutions.


res_dict = {
    # 'resource': 'local.localhost',
    'resource': 'princeton.traverse',
    'project_id': 'test',
    'schema': 'local',
    'walltime': 45,
    'cpus': 32 * 1,
    'gpus': 4 * 1,
}

appman = AppManager(hostname=hostname, port=port, resubmit_failed=False,
                    username=username, password=password)
appman.resource_desc = res_dict
appman.workflow = set([p])
appman.run()
