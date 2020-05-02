#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import nested_scopes, generators, division, absolute_import, \
    with_statement, print_function

import os
import sys
import yaml
import math
import logging
import argparse
from radical.entk import Pipeline, Stage, Task, AppManager

# Local stuff
from gcmt3d.utils.io import get_location_in_database
from gcmt3d.utils.io import get_cmt_id
from gcmt3d.log_util import modify_logger
from gcmt3d.utils.io import read_yaml_file

logger = logging.getLogger('gcmt3d')
modify_logger(logger)

# ------------------------------------------------------------------------------
# Set default verbosity

if os.environ.get("RADICAL_ENTK_VERBOSE") == None:
    os.environ["RADICAL_ENTK_REPORT"] = "True"

os.environ["RADICAL_VERBOSE"] = "DEBUG"
os.environ["RADICAL_ENTK_VERBOSE"] = "DEBUG"

# Description of how the RabbitMQ process is accessible
# No need to change/set any variables if you installed RabbitMQ has a system
# process. If you are running RabbitMQ under a docker container or another
# VM, set "RMQ_HOSTNAME" and "RMQ_PORT" in the session where you are running
# this script.
hostname = os.environ.get("RMQ_HOSTNAME", "localhost")
port = int(os.environ.get("RMQ_PORT", 5672))

# DEFINES WHETHER THE HEADNODE IS AVAILABLE FOR DOWNLOAD FROM WITHIN AN ENTK
# WORFLOW
HEADNODE_AVAILABLE = False

CMT_LIST = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
            "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]


def write_sources(cmt_file_db, param_path, task_counter):
    """ This function creates a stage that modifies the CMTSOLUTION files
    before the simulations are run.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param task_counter: total task count up until now in pipeline
    :return: EnTK Stage

    """

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    # Database parameters.
    dbparams = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and Cid
    cdir = os.path.dirname(cmt_file_db)
    cid = get_cmt_id(cmt_file_db)

    # Create a Stage object
    w_sources = Stage()
    w_sources.name = "Write-Sources"

    # Create Task for stage
    w_sources_t = Task()
    w_sources_t.name = "And-Stations"
    w_sources.pre_exec = [
        dbparams["anaconda"],
        dbparams["conda-sh"]
    ]
    print("writetask")
    print(w_sources.pre_exec)
    w_sources_t.executable = ["write-sources"]
    w_sources_t.arguments = ['-f', cmt_file_db,
                             '-p', param_path]

    w_sources_t.cpu_reqs = {
            'processes': 1,
            'process_type': None,
            'threads_per_process': 1,
            'thread_type': None}

    # In the future maybe to database dir as a total log?
    w_sources_t.stdout = os.path.join(cdir, "logs",
                                      "stdout.pipeline_%s.task_%s.%s"
                                      % (cid, str(task_counter).zfill(4),
                                         w_sources_t.name))
    w_sources_t.stderr = os.path.join(cdir, "logs",
                                      "stderr.pipeline_%s.task_%s.%s"
                                      % (cid, str(task_counter).zfill(4),
                                         w_sources_t.name))

    # Add Task to the Stage
    w_sources.add_tasks(w_sources_t)

    task_counter += 1

    return w_sources, task_counter


def run_specfem(cmt_file_db, param_path, task_counter):
    """ This function runs the necessary Specfem simulations.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param task_counter: total task count up until now in pipeline
    :return: EnTK Stage

    """

    # Earthquake specific database parameters: Dir and Cid
    cdir = os.path.dirname(cmt_file_db)
    cid = get_cmt_id(cmt_file_db)

    specfemspec_path = os.path.join(param_path,
                                    "SpecfemParams/SpecfemParams.yml")
    comp_and_modules_path = os.path.join(param_path,
                                         "SpecfemParams/"
                                         "CompilersAndModules.yml")

    # The simulation directories.
    attr = CMT_LIST

    # Load Parameters
    specfemspecs = read_yaml_file(specfemspec_path)
    cm_dict = read_yaml_file(comp_and_modules_path)

    # Simulation directory
    simdir = os.path.join(cdir, "CMT_SIMs")

    # Create a Stage object
    runSF3d = Stage()
    runSF3d.name = "Simulation"

    for at in attr[:2]:
        sf_t = Task()
        sf_t.name = "run-" + at

        # Module Loading
        sf_t.pre_exec = [  # Get rid of existing modules
                           "module purge"]
        # Append to pre_execution module list.
        for module in cm_dict["modulelist"]:
            sf_t.pre_exec.append("module load %s" % module)

        # Change directory to specfem directories
        sf_t.pre_exec.append(  # Change directory
            "cd %s" % os.path.join(simdir, at))
        print("simtask:", at)
        print(sf_t.pre_exec)
        sf_t.executable = ["./bin/xspecfem3D"]  # Assigned executable

        # Resource assignment
        sf_t.cpu_reqs = {'processes': 6, 'process_type': 'MPI',
                         'threads_per_process': 1, 'thread_type': 'OpenMP'}

        # Note that it's one GPU per MPI task
        sf_t.gpu_reqs = {'processes': 1, 'process_type': 'MPI',
                         'threads_per_process': 1, 'thread_type': 'CUDA'}

        # In the future maybe to database dir as a total log?
        sf_t.stdout = os.path.join("%s" % cdir, "logs",
                                   "stdout.pipeline_%s.task_%s.%s"
                                   % (cid,
                                      str(task_counter).zfill(4),
                                      sf_t.name))

        sf_t.stderr = os.path.join("%s" % cdir, "logs",
                                   "stderr.pipeline_%s.task_%s.%s"
                                   % (cid,
                                      str(task_counter).zfill(4),
                                      sf_t.name))


        # Increase Task counter
        task_counter += 1

        # Add Task to the Stage
        runSF3d.add_tasks(sf_t)

    return runSF3d, task_counter


def convert_traces(cmt_file_db, param_path, task_counter):
    """This function creates the to-ASDF conversion stage. Meaning, in this
    stage, both synthetic and observed traces are converted to ASDF files.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param pipelinedir: path to pipeline directory
    :return: EnTK Stage

    """

    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Parameters
    dbparams = read_yaml_file(databaseparam_path)

    # File and directory
    cid = get_cmt_id(cmt_file_db)
    cmt_dir = os.path.dirname(cmt_file_db)
    sim_dir = os.path.join(cmt_dir, "CMT_SIMs")


    ## Create a Stage object
    conversion_stage = Stage()
    conversion_stage.name = "Convert"

    attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
            "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

    ##### Converting the synthetic data
    logger.verbose(" ")
    logger.verbose("Converting synthetic traces to ASDF ... ")
    logger.verbose(" ")

    # for _i, at in enumerate(attr[:dbparams["npar"] + 1]):
    for _i, at in enumerate(attr[:2]):

        # Path file
        syn_path_file = os.path.join(sim_dir, at, at + ".yml")

        # Create Task for stage
        c_task = Task()
        c_task.name = at

        c_task.pre_exec = [dbparams["anaconda"],
                           dbparams["conda-sh"]]
        print("ctask:", at)
        print(c_task.pre_exec)
        c_task.executable = ["convert2asdf"]

        arguments = ["-f", syn_path_file]
        if logger.getEffectiveLevel() < logging.INFO:
            arguments.append("-v")

        c_task.arguments = arguments

        # In the future maybe to database dir as a total log?
        c_task.stdout = os.path.join("%s" % cmt_dir, "logs",
                                          "stdout.pipeline_%s.task_%s.%s"
                                          % (cid, str(task_counter).zfill(4),
                                             c_task.name))

        c_task.stderr = os.path.join("%s" % cid, "logs",
                                          "stderr.pipeline_%s.task_%s.%s"
                                          % (cid, str(task_counter).zfill(4),
                                             c_task.name))

        # Increase Task counter
        task_counter += 1

        conversion_stage.add_tasks(c_task)

    ##### Converting the observed data
    logger.verbose("\nConverting observed traces to ASDF ... \n")

    obs_path_file = os.path.join(cmt_dir, "seismograms", "obs", "observed.yml")

    # Create Task for stage
    c_task = Task()
    c_task.name = "Observed"

    c_task.pre_exec = [dbparams["anaconda"],
                       dbparams["conda-sh"]]

    c_task.executable = "convert2asdf"

    arguments = ["-f", obs_path_file]
    if logger.getEffectiveLevel() < logging.INFO:
        arguments.append("-v")

    c_task.arguments = arguments

    # In the future maybe to database dir as a total log?
    c_task.stdout = os.path.join("%s" % cmt_dir, "logs",
                                 "stdout.pipeline_%s.task_%s.%s"
                                 % (cid, str(task_counter).zfill(4),
                                    c_task.name))

    c_task.stderr = os.path.join("%s" % cmt_dir, "logs",
                                 "stderr.pipeline_%s.task_%s.%s"
                                 % (cid, str(task_counter).zfill(4),
                                    c_task.name))

    # Increase Task counter
    task_counter += 1

    conversion_stage.add_tasks(c_task)

    return conversion_stage, task_counter


def workflow(cmtfilenames, param_path):
    """This function submits the complete workflow

    :param cmt_filename: str containing the path to the cmt solution that is
                      supposed to be inverted for

    Usage:
        ```bash
        python 1pipeline <path/to/cmtsolution>
        ```

    """
    # Just get the full path
    param_path = os.path.abspath(param_path)

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = read_yaml_file(databaseparam_path)

    # List of pipelines
    pipe_list = []

    # Create pipe lines for all jobss
    for _cmtfile in cmtfilenames:

        # Earthquake specific database parameters: Dir and Cid
        cmt_in_db = get_location_in_database(_cmtfile,
                                             DB_params["entkdatabase"])

        # Create a counter for all tasks in one pipeline
        task_counter = 0

        # Create a Pipeline object
        p = Pipeline()

        # ---- Write Sources ---------------------------------------------------- #

        # Create Source modification stage
        w_sources_stage, task_counter = write_sources(cmt_in_db, param_path,
                                                      task_counter)

        # Add Stage to the Pipeline
        p.add_stages(w_sources_stage)

        # ---- Run Specfem -------------------------------------------------- #

        # Create Specfem Stage
        runSF3D_stage, task_counter = run_specfem(cmt_in_db,
                                                  param_path,
                                                  task_counter)

        # Add Simulation stage to the Pipeline
        p.add_stages(runSF3D_stage)

        # ---- Convert to ASDF ---------------------------------------------- #

        # Create conversion stage
        conversion_stage, task_counter = convert_traces(cmt_in_db, param_path,
                                                        task_counter)

        # Add stage to pipeline
        p.add_stages(conversion_stage)


        pipe_list.append(p)
    # ============== RUNNING THE PIPELINE =================================== #

    # Create Application Manager
    appman = AppManager(hostname=hostname, port=port,
                        resubmit_failed=False)

    # # Compute the necessary walltime from walltime/per simulation
    # # Load parameters
    # specfem_specs = read_yaml_file(
    #     os.path.join(param_path, "SpecfemParams/SpecfemParams.yml"))
    #
    # # Get twalltime from walltime specification in the parameter file.
    # walltime_per_simulation = specfem_specs["walltime"].split(":")
    # hours_in_min = float(walltime_per_simulation[0])*60
    # min_in_min = float(walltime_per_simulation[1])
    # sec_in_min = float(walltime_per_simulation[2])/60
    #
    # cpus = int(specfem_specs["cpus"])
    # tasks = int(specfem_specs["tasks"])
    #
    # # Add times to get full simulation time. The 45 min are accounting for
    # # everything that is not simulation time
    # total_min = int(1/math.ceil(float(cpus)/40) \
    #     * 10 * int(round(hours_in_min + min_in_min + sec_in_min)) + 45)

    # Create a dictionary describe four mandatory keys:
    # resource, walltime, cpus etc.
    # resource is "local.localhost" to execute locally
    # Define which resources to get depending on how specfem is run!

    hour = 3600
    res_dict = {
        'resource': 'princeton.traverse',
        'project': 'geo',
        'schema': 'local',
        'walltime': int(0.6*hour),
        'cpus': int(2*6*4),
        'gpus': 12
    }

    # Assign resource request description to the Application Manager
    appman.resource_desc = res_dict

    # Assign the workflow as a set or list of Pipelines to the Application Manager
    # Note: The list order is not guaranteed to be preserved
    appman.workflow = set(pipe_list)

    # Run the Application Manager
    appman.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmtfiles", help="Path to CMTSOLUTION file",
                        type=str, nargs="+")
    parser.add_argument("-p", dest="param_path", type=str, required=True,
                        help="Path to workflow paramater directory")
    args = parser.parse_args()

    # Make file list either way
    if type(args.cmtfiles) is str:
        cmtfiles = [args.cmtfiles]
    else:
        cmtfiles = args.cmtfiles

    # Run
    workflow(cmtfiles, args.param_path)
