#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import nested_scopes, generators, division, absolute_import, \
    with_statement, print_function

from radical.entk import Pipeline, Stage, Task, AppManager
import os
import argparse
from _get_eq_dir import get_eq_entry_path

# Hopefully this changes soon with entk for python 3
import sys
sys.path.append(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # GCMT3D DIR
    "gcmt3d", "data", "management"))
from create_process_paths import get_windowing_list
from create_process_paths import get_processing_list
import yaml
import math

# For calling binaries without entkd
import subprocess
from shlex import split

bin_path = os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))),
                        "bins")

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


def read_yaml_file(filename):
    """read yaml file"""
    with open(filename) as fh:
        return yaml.load(fh, Loader=yaml.Loader)


def create_entry(cmt_filename, param_path, pipelinedir, task_counter):
    """This function creates the Entk stage for creation of a database entry.

    :param cmt_filename: cmt_filename
    :param param_path: parameter directory
    :param pipelinedir: Directory of the pipeline
    :return: EnTK Stage
    """

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_filename)

    # Path to function
    create_database_func = os.path.join(bin_path, "create_entry.py")

    # Create a Stage object
    database_entry = Stage()

    t1 = Task()
    t1.name = "database-entry"
    t1.pre_exec = [  # Conda activate
                     DB_params["conda-activate"]]
    t1.executable = DB_params["bin-python"]  # Assign
    # executable to the task
    t1.arguments = [create_database_func,
                   os.path.abspath(cmt_filename)]

    # In the future maybe to database dir as a total log?
    t1.stdout = os.path.join(pipelinedir, "database-entry." + eq_id + ".stdout")
    t1.stderr = os.path.join(pipelinedir, "database-entry." + eq_id + ".stderr")

    # In the future maybe to database dir as a total log?
    t1.stdout = os.path.join("%s" % eq_dir, "logs",
                             "stdout.pipeline_%s.task_%s.%s"
                             % (eq_id, str(task_counter).zfill(4),
                                t1.name))

    t1.stderr = os.path.join("%s" % eq_dir, "logs",
                             "stderr.pipeline_%s.task_%s.%s"
                             % (eq_id, str(task_counter).zfill(4),
                                t1.name))

    # Add Task to the Stage
    database_entry.add_tasks(t1)

    return database_entry


def call_create_entry(cmt_filename, param_path, pipelinedir, task_counter):
    """Simply calls the binary to create an entry without making it a stage.
    Hence, it would be run prior to the pipeline start.

    :param cmt_filename: cmt_filename from wherever
    :param param_path: path to parameter files
    :param pipelinedir: Directory of the pipeline
    :return: nothing as it is simply a function call.
    """

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_filename)

    # Path to function
    create_database_func = os.path.join(bin_path, "create_entry.py")

    if DB_params["verbose"]:
        print("Creating the entry outside of the pipeline!")

    # Create command -N nodes, -n tasks, -D change directory
    bash_command = "%s\n %s %s %s\n %s" \
                       % (DB_params["conda-activate"],
                          DB_params["bin-python"],
                          create_database_func,
                          cmt_filename,
                          DB_params["conda-deactivate"])

    create_entry_t = "database-entry"

    # In the future maybe to database dir as a total log?
    stdout = os.path.join("%s" % eq_dir, "logs",
                          "stdout.pipeline_%s.task_%s.%s"
                          % (eq_id, str(task_counter).zfill(4),
                             create_entry_t))

    stderr = os.path.join("%s" % eq_dir, "logs",
                          "stderr.pipeline_%s.task_%s.%s"
                          % (eq_id, str(task_counter).zfill(4),
                             create_entry_t))

    if DB_params["verbose"]:
        # Send command
        subprocess.check_output(bash_command, shell=True)

    else:
        # Send command
        with open(stdout, "wb") as out, open(stderr, "wb") as err:
            subprocess.check_output(bash_command, shell=True, stderr=err)



def call_download_data(cmt_file_db, param_path, pipelinedir, task_counter):
    """Simply calls the binary to download the observed data.

    :param cmt_file_db: cmt_file in the database
    :param param_path: path to parameter files
    :param pipelinedir: Directory of the pipeline
    :return: nothing as it is simply a function call.
    """

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_file_db)

    # Path to function
    download_data_func = os.path.join(bin_path, "request_data.py")

    if DB_params["verbose"]:
        print("Download outside pipeline!")

    # Create command -N nodes, -n tasks, -D change directory
    bash_command = "%s; %s %s %s; %s" \
                   % (DB_params["conda-activate"],
                      DB_params["bin-python"],
                      download_data_func,
                      cmt_file_db,
                      DB_params["conda-deactivate"])

    datarequest_t = "data-request"

    # In the future maybe to database dir as a total log?
    stdout = os.path.join("%s" % eq_dir, "logs",
                                        "stdout.pipeline_%s.task_%s.%s"
                                        % (eq_id, str(task_counter).zfill(4),
                                           datarequest_t))

    stderr = os.path.join("%s" % eq_dir, "logs",
                                        "stderr.pipeline_%s.task_%s.%s"
                                        % (eq_id, str(task_counter).zfill(4),
                                           datarequest_t))

    if DB_params["verbose"]:
        # Send command
        subprocess.check_output(bash_command, shell=True)
    else:
        # Send command
        with open(stdout, "wb") as out, open(stderr, "wb") as err:
            subprocess.check_output(bash_command, shell=True, stderr=err)


def data_request(cmt_file_db, param_path, pipelinedir, task_counter):
    """ This function creates the request for the observed data and returns
    it as an EnTK Stage

    :param cmt_file_db: cmt_file in the database
    :param param_path: path to parameter file directory
    :param pipeline_dir: pipeline directory
    :return: EnTK Stage

    """

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_file_db)

    # # Path to function
    request_data_func = os.path.join(bin_path, "request_data.py")

    # Create a Stage object
    datarequest = Stage()

    datarequest_t = Task()
    datarequest_t.name = "data-request"
    datarequest_t.pre_exec = [  # Conda activate
                                DB_params["conda-activate"]]
    datarequest_t.executable = DB_params["bin-python"]  # Assign executable
                                                          # to the task
    datarequest_t.arguments = [request_data_func, cmt_file_db]

    # In the future maybe to database dir as a total log?
    datarequest_t.stdout = os.path.join("%s" % eq_dir, "logs",
                                        "stdout.pipeline_%s.task_%s.%s"
                                        % (eq_id, str(task_counter).zfill(4),
                                           datarequest_t.name))

    datarequest_t.stderr = os.path.join("%s" % eq_dir, "logs",
                                        "stderr.pipeline_%s.task_%s.%s"
                                        % (eq_id, str(task_counter).zfill(4),
                                           datarequest_t.name))

    # Add Task to the Stage
    datarequest.add_tasks(datarequest_t)

    return datarequest


def write_sources(cmt_file_db, param_path, pipelinedir, task_counter):
    """ This function creates a stage that modifies the CMTSOLUTION files
    before the simulations are run.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param pipelinedir: path to pipeline directory
    :return: EnTK Stage

    """

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_file_db)

    # Path to function
    write_source_func = os.path.join(bin_path, "write_sources.py")

    # Create a Stage object
    w_sources = Stage()

    w_sources.name = "Write-Sources"

    # Create Task for stage
    w_sources_t = Task()
    w_sources_t.name = "Task-Sources"
    w_sources_t.pre_exec = [  # Conda activate
                              DB_params["conda-activate"]]
    w_sources_t.executable = DB_params["bin-python"]  #
        # Assign executable
        # to the task
    w_sources_t.arguments = [write_source_func, cmt_file_db]

    # In the future maybe to database dir as a total log?
    w_sources_t.stdout = os.path.join("%s" % eq_dir, "logs",
                                      "stdout.pipeline_%s.task_%s.%s"
                                      % (eq_id, str(task_counter).zfill(4),
                                         w_sources_t.name))

    w_sources_t.stderr = os.path.join("%s"% eq_dir, "logs",
                                      "stderr.pipeline_%s.task_%s.%s"
                                      % (eq_id, str(task_counter).zfill(4),
                                         w_sources_t.name))

    # Add Task to the Stage
    w_sources.add_tasks(w_sources_t)

    task_counter += 1

    return w_sources, task_counter


def run_specfem(cmt_file_db, param_path, pipelinedir):
    """ This function runs the necessary Specfem simulations.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param pipelinedir: path to pipeline directory
    :return: EnTK Stage

    """

    specfemspec_path = os.path.join(param_path,
                                    "SpecfemParams/SpecfemParams.yml")
    comp_and_modules_path = os.path.join(param_path,
                                         "SpecfemParams/"
                                         "CompilersAndModules.yml")

    # Load Parameters
    specfemspecs = read_yaml_file(specfemspec_path)
    cm_dict = read_yaml_file(comp_and_modules_path)

    attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
            "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

    simdir = os.path.join(os.path.dirname(cmt_file_db), "CMT_SIMs")

    # Create a Stage object
    runSF3d = Stage()
    runSF3d.name = "Simulation"

    for at in attr[0]:
        sf_t = Task()
        sf_t.name = "run-" + at

        # Module Loading
        sf_t.pre_exec = [  # Get rid of existing modules
            "module purge"]
        for module in cm_dict["modulelist"]:
            sf_t.pre_exec.append("module load %s" % module)
        sf_t.pre_exec.append("module load %s" % cm_dict["gpu_module"])

        # Change directory to specfem directories
        sf_t.pre_exec.append(  # Change directory
            "cd %s" % os.path.join(simdir, at))

        sf_t.executable = "./bin/xspecfem3D"  # Assign executable

        # In the future maybe to database dir as a total log?
        sf_t.stdout = os.path.join(pipelinedir,
                                   "run_specfem." + cmt_file_db[3:-4] +
                                   ".stdout")
        sf_t.stderr = os.path.join(pipelinedir,
                                   "run_specfem." + cmt_file_db[3:-4] +
                                   ".stderr")

        sf_t.cpu_reqs = {
            "processes": 54,
            "process_type": "MPI",
            "threads_per_process": 1,
            "thread_type": None
        }

        # Add Task to the Stage
        runSF3d.add_tasks(sf_t)

        return runSF3d


def specfem_clean_up(cmt_file_db, param_path, pipelinedir):
    """ Cleaning up the simulation directories since we don"t need all the
    files for the future.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param pipelinedir: path to pipeline directory
    :return: EnTK Stage

    """

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    DB_params = read_yaml_file(databaseparam_path)

    # Path to function
    clean_up_func = os.path.join(bin_path, "clean_up_simdirs.py")

    # Create a Stage object
    clean_up = Stage()
    clean_up.name = "Clean-Up"

    # Create Task for stage
    clean_up_t = Task()
    clean_up_t.name = "Task-Clean-Up"
    clean_up_t.pre_exec = [  # Conda activate
        DB_params["conda-activate"]]
    clean_up_t.executable = DB_params["bin-python"]  # Assign executable
    # to the task
    clean_up_t.arguments = [clean_up_func, cmt_file_db]

    # In the future maybe to database dir as a total log?
    clean_up_t.stdout = os.path.join(pipelinedir,
                                      "clean_up." + cmt_file_db[3:-4] +
                                      ".stdout")
    clean_up_t.stderr = os.path.join(pipelinedir,
                                      "clean_up." + cmt_file_db[3:-4] +
                                      ".stderr")

    # Add Task to the Stage
    clean_up.add_tasks(clean_up_t)

    return clean_up


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
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_file_db)

    # File and directory
    cmt_dir = os.path.dirname(cmt_file_db)
    sim_dir = os.path.join(cmt_dir, "CMT_SIMs")


    ## Create a Stage object
    conversion_stage = Stage()
    conversion_stage.name = "Convert"

    # Conversion binary
    conversion_bin = os.path.join(bin_path, "convert_to_asdf.py")


    attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
            "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

    ##### Converting the synthetic data
    if DB_params["verbose"]:
        print("\nConverting synthetic traces to ASDF ... \n")

    for _i, at in enumerate(attr[:DB_params["npar"] + 1]):

        # Path file
        syn_path_file = os.path.join(sim_dir, at, at + ".yml")

        # Create Task for stage
        c_task = Task()
        c_task.name = at

        c_task.pre_exec = [DB_params["conda-activate"]]
        c_task.executable = DB_params["bin-python"]  # Assign executable
                                                       # to the task

        arguments = [conversion_bin, "-f", syn_path_file]
        if DB_params["verbose"]:
            arguments.append("-v")

        c_task.arguments = arguments

        # In the future maybe to database dir as a total log?
        c_task.stdout = os.path.join("%s" % eq_dir, "logs",
                                          "stdout.pipeline_%s.task_%s.%s"
                                          % (eq_id, str(task_counter).zfill(4),
                                             c_task.name))

        c_task.stderr = os.path.join("%s" % eq_dir, "logs",
                                          "stderr.pipeline_%s.task_%s.%s"
                                          % (eq_id, str(task_counter).zfill(4),
                                             c_task.name))

        # Increase Task counter
        task_counter += 1

        conversion_stage.add_tasks(c_task)


    ##### Converting the observed data
    if DB_params["verbose"]:
        print("\nConverting observed traces to ASDF ... \n")

    obs_path_file = os.path.join(cmt_dir, "seismograms", "obs", "observed.yml")

    # Create Task for stage
    c_task = Task()
    c_task.name = "Observed"

    c_task.pre_exec = [DB_params["conda-activate"]]
    c_task.executable = DB_params["bin-python"]  # Assign executable
    # to the task

    # Create Argument list
    arguments = [conversion_bin, "-f", obs_path_file]
    if DB_params["verbose"]:
        arguments.append("-v")

    c_task.arguments = arguments

    # In the future maybe to database dir as a total log?
    c_task.stdout = os.path.join("%s" % eq_dir, "logs",
                                 "stdout.pipeline_%s.task_%s.%s"
                                 % (eq_id, str(task_counter).zfill(4),
                                    c_task.name))

    c_task.stderr = os.path.join("%s" % eq_dir, "logs",
                                 "stderr.pipeline_%s.task_%s.%s"
                                 % (eq_id, str(task_counter).zfill(4),
                                    c_task.name))
    # Increase Task counter
    task_counter += 1

    conversion_stage.add_tasks(c_task)

    return conversion_stage, task_counter


def create_process_path_files(cmt_file_db, param_path, pipelinedir,
                              task_counter):
    """This function creates the path files used for processing both
    synthetic and observed data in ASDF format, as well as the following
    windowing procedure.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param pipelinedir: path to pipeline directory
    :return: EnTK Stage

    """

    # Get database parameter path
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Parameters
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_file_db)

    # Process path function
    create_process_path_bin = os.path.join(bin_path,
                                           "create_path_files.py")

    # Create Process Paths Stage (CPP)
    # Create a Stage object
    cpp = Stage()
    cpp.name = "CreateProcessPaths"

    # Create Task
    cpp_t = Task()
    cpp_t.name = "CPP-Task"
    cpp_t.pre_exec = [  # Conda activate
                      DB_params["conda-activate"]]
    cpp_t.executable = DB_params["bin-python"]  # Assign executable
                                                  # to the task
    cpp_t.arguments = [create_process_path_bin, cmt_file_db]

    # In the future maybe to database dir as a total log?
    cpp_t.stdout = os.path.join("%s" % eq_dir, "logs",
                              "stdout.pipeline_%s.task_%s.%s"
                              % (eq_id, str(task_counter).zfill(4),
                                 cpp_t.name))

    cpp_t.stderr = os.path.join("%s" % eq_dir, "logs",
                              "stderr.pipeline_%s.task_%s.%s"
                              % (eq_id, str(task_counter).zfill(4),
                                 cpp_t.name))

    task_counter += 1

    cpp.add_tasks(cpp_t)

    return cpp, task_counter


def create_processing_stage(cmt_file_db, param_path, task_counter):
    """This function creates the ASDF processing stage.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param pipelinedir: path to pipeline directory
    :return: EnTK Stage

    """

    # Get database parameter path
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Parameters
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_file_db)

    # Processing param dir
    process_obs_param_dir = os.path.join(param_path, "ProcessObserved")
    process_syn_param_dir = os.path.join(param_path, "ProcessSynthetic")

    # Process path list
    # Important step! This creates a processing list prior to having created
    # the actual process path files. It is tested so it definitely works!
    # this way the processes can be distributed for each ASDF file on one
    # processor or more (MPI enabled!)
    processing_list, _, _ = get_processing_list(cmt_file_db,
                                                process_obs_param_dir,
                                                process_syn_param_dir,
                                                verbose=True)

    # Process path function
    process_func = os.path.join(bin_path, "process_asdf.py")

    # Create Process Paths Stage (CPP)
    # Create a Stage object
    process_stage = Stage()
    process_stage.name = "Processing"

    # Number of Processes:
    N = len(processing_list)

    # Loop over process path files
    for process_path in processing_list:

        # Create Task
        processing_task = Task()

        # This way the task gets the name of the path file
        processing_task.name = "Processing-" \
                               + os.path.basename(process_path)

        processing_task.pre_exec = [  # Conda activate
                                      DB_params["conda-activate"]]

        processing_task.executable = DB_params["bin-python"]  # Assign exec.
                                                                # to the task

        # Create Argument list
        arguments = [process_func, "-f", process_path]
        if DB_params["verbose"]:
            arguments.append("-v")

        processing_task.arguments = arguments

        # In the future maybe to database dir as a total log?
        processing_task.stdout = os.path.join("%s" % eq_dir, "logs",
                                  "stdout.pipeline_%s.task_%s.%s"
                                  % (eq_id, str(task_counter).zfill(4),
                                     processing_task.name))

        processing_task.stderr = os.path.join("%s" % eq_dir, "logs",
                                  "stderr.pipeline_%s.task_%s.%s"
                                  % (eq_id, str(task_counter).zfill(4),
                                     processing_task.name))

        task_counter += 1

        process_stage.add_tasks(processing_task)

    return process_stage, task_counter


def create_windowing_stage(cmt_file_db, param_path, task_counter):
    """This function creates the ASDF windowing stage.

    :param cmt_file_db: cmtfile in the database
    :param param_path: path to parameter file directory
    :param pipelinedir: path to pipeline directory
    :return: EnTK Stage

    """

    # Get database parameter path
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Parameters
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_file_db)

    # Windowing parameter file directory
    window_process_dir = os.path.join(param_path, "CreateWindows")

    # Window path list
    # Important step! This creates a windowing list prior to having created
    # the actual window path files. It is tested so it definitely works!
    # This way the windowing processes can be distributed for each ASDF file
    # pair on one processor (No MPI support!)

    window_path_list, _ = get_windowing_list(cmt_file_db, window_process_dir,
                                             verbose=False)

    # Process path function
    window_func = os.path.join(bin_path, "window_selection_asdf.py")

    # Create Process Paths Stage (CPP)
    # Create a Stage object
    window_stage = Stage()
    window_stage.name = "Windowing"

    # Loop over process path files
    for window_path in window_path_list:

        # Create Task
        window_task = Task()

        # This way the task gets the name of the path file
        window_task.name = os.path.basename(window_path)

        window_task.pre_exec = [  # Conda activate
                                  DB_params["conda-activate"]]

        window_task.executable = [DB_params["bin-python"]]  # Assign exec
                                                            # to the task

        # Create Argument list
        arguments = [window_func, "-f", window_path]
        if DB_params["verbose"]:
            arguments.append("-v")

        window_task.arguments = arguments

        # In the future maybe to database dir as a total log?
        window_task.stdout = os.path.join("%s" % eq_dir, "logs",
                                              "stdout.pipeline_%s.task_%s.%s"
                                              % (
                                              eq_id, str(task_counter).zfill(4),
                                              window_task.name))

        window_task.stderr = os.path.join("%s" % eq_dir, "logs",
                                              "stderr.pipeline_%s.task_%s.%s"
                                              % (
                                              eq_id, str(task_counter).zfill(4),
                                              window_task.name))

        window_stage.add_tasks(window_task)

    return window_stage, task_counter


def create_inversion_dict_stage(cmt_file_db, param_path):
    """Creates stage for the creation of the inversion files. This stage is
    tiny, but required before the actual inversion.

    :param cmt_file_db:
    :param param_path:
    :return:
    """

    # Get database parameter path
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Parameters
    DB_params = read_yaml_file(databaseparam_path)

    # Function
    inv_dict_func = os.path.join(bin_path, "write_inversion_dicts.py")

    # Create Process Paths Stage (CPP)
    # Create a Stage object
    inv_dict_stage = Stage()
    inv_dict_stage.name = "Creating"

    # Create Task
    inv_dict_task = Task()

    # This way the task gets the name of the path file
    inv_dict_task.name = "Inversion-Dictionaries"

    inv_dict_task.pre_exec = [  # Conda activate
                              DB_params["conda-activate"]]

    inv_dict_task.executable = [DB_params["bin-python"]]  # Assign exec
    # to the task

    inv_dict_task.arguments = [inv_dict_func,
                               "-f", cmt_file_db,
                               "-p", param_path]

    inv_dict_stage.add_tasks(inv_dict_task)

    return inv_dict_stage


def inversion_stage(cmt_file_db, param_path):
    """Creates inversion stage.

    :param cmt_file_db:
    :param param_path:
    :return:
    """

    # Get database parameter path
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")

    # Load Parameters
    DB_params = read_yaml_file(databaseparam_path)

    # Function
    inversion_func = os.path.join(bin_path, "inversion.py")

    # Create Process Paths Stage (CPP)
    # Create a Stage object
    inversion_stage = Stage()
    inversion_stage.name = "CMT3D"

    # Create Task
    inversion_task = Task()

    # This way the task gets the name of the path file
    inversion_task.name = "Inversion"

    inversion_task.pre_exec = [  # Conda activate
        DB_params["conda-activate"]]

    inversion_task.executable = DB_params["bin-python"]  # Assign exec
                                                           # to the task

    inversion_task.arguments = [inversion_func,
                                "-f", cmt_file_db,
                                "-p", param_path]

    inversion_stage.add_tasks(inversion_task)

    return inversion_stage


def workflow(cmt_filename):
    """This function submits the complete workflow

    :param cmt_filename: str containing the path to the cmt solution that is
                      supposed to be inverted for

    Usage:
        ```bash
        python pipeline <path/to/cmtsolution>
        ```

    """

    # Task & pipeline counter
    pipeline_counter = 0
    task_counter = 0

    # Path to pipeline file
    pipelinepath = os.path.abspath(__file__)
    pipelinedir = os.path.dirname(pipelinepath)

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(pipelinedir), "params")

    # Get Database parameters
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = read_yaml_file(databaseparam_path)

    # Earthquake specific database parameters: Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_filename)

    # Earthquake file in the database
    cmt_file_db = os.path.join(eq_dir, "eq_" + eq_id + ".cmt")


    # Create a Pipeline object
    p = Pipeline()

    if HEADNODE_AVAILABLE:
        # ---- Create Database Entry --------------------------------------------- #

        # Create Database entry stage:
        database_entry_stage = create_entry(cmt_filename, param_path,
                                            pipelinedir, task_counter)

        # Add Stage to the Pipeline
        p.add_stages(database_entry_stage)

        # ---- REQUEST DATA ------------------------------------------------------ #

        # Request data stage
        datarequest_stage = data_request(cmt_file_db, param_path,
                                         pipelinedir, task_counter)

        # Add Stage to the Pipeline
        p.add_stages(datarequest_stage)
    else:

        # Create the entry now before running the pipeline
        call_create_entry(cmt_filename, param_path, pipelinedir, task_counter)

        # # Download the data from the headnode before running the pipeline
        # call_download_data(cmt_file_db, param_path, pipelinedir, task_counter)

    # ---- Write Sources ----------------------------------------------------- #

    # Create Source modification stage
    w_sources_stage, task_counter = write_sources(cmt_file_db, param_path,
                                                  pipelinedir,
                                                  task_counter)

    # Add Stage to the Pipeline
    p.add_stages(w_sources_stage)

    # ---- Run Specfem ------------------------------------------------------- #

    # # Create Specfem Stage
    # runSF3D_stage = run_specfem(cmt_file_db, param_path, pipelinedir)
    #
    # # Add Simulation stage to the Pipeline
    # p.add_stages(runSF3D_stage)
    #
    # # ---- Clean Up Specfem -------------------------------------------------- #
    #
    # # Create clean_up stage
    # clean_up_stage = specfem_clean_up(cmt_file_db, param_path, pipelinedir)
    #
    # # Add Stage to the Pipeline
    # p.add_stages(clean_up_stage)

    # ---- Convert to ASDF --------------------------------------------------- #

    # Create conversion stage
    conversion_stage, task_counter = convert_traces(cmt_file_db, param_path,
                                                    task_counter)

    # Add stage to pipeline
    p.add_stages(conversion_stage)

    # ---- Create Process Path files ----------------------------------------- #

    # Create Process Stage Pipeline
    process_path_stage, task_counter = create_process_path_files(cmt_file_db,
                                                                 param_path,
                                                                 pipelinedir,
                                                                 task_counter)

    p.add_stages(process_path_stage)

    # ---- Process Traces ---------------------------------------------------- #

    # Create processing stage
    processing_stage, task_counter = create_processing_stage(cmt_file_db,
                                                             param_path,
                                                             task_counter)

    p.add_stages(processing_stage)

    # # ---- Window Traces ---------------------------------------------------- #

    # Create processing stage
    windowing_stage, task_counter = create_windowing_stage(cmt_file_db,
                                                           param_path,
                                                           task_counter)

    p.add_stages(windowing_stage)

    # ============== RUNNING THE PIPELINE ==================================== #

    # Create Application Manager
    appman = AppManager(hostname=hostname, port=port)

    # Create a dictionary describe four mandatory keys:
    # resource, walltime, and cpus
    # resource is "local.localhost" to execute locally
    # res_dict_gpu = {
    #     "resource": "princeton.tiger_gpu",
    #     "project": "geo",
    #     "queue": "gpu",
    #     "schema": "local",
    #     "walltime": 300,
    #     "cpus": 6,
    #     "gpus": 6
    # }

    res_dict_cpu = {
        "resource": "princeton.tiger_cpu",
        "project": "geo",
        "queue": "cpu",
        "schema": "local",
        "walltime": 30,
        "cpus": 10,
    }

    # Assign resource request description to the Application Manager
    appman.resource_desc = res_dict_cpu

    # Assign the workflow as a set or list of Pipelines to the Application Manager
    # Note: The list order is not guaranteed to be preserved
    appman.workflow = set([p])

    # Run the Application Manager
    appman.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to CMTSOLUTION file",
                        type=str)
    args = parser.parse_args()

    # Run
    workflow(args.filename)
