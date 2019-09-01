from radical.entk import Pipeline, Stage, Task, AppManager
import os
import argparse
from ._get_eq_dir import get_eq_entry_path
from gcmt3d.data.management.create_process_paths import get_windowing_list
from gcmt3d.data.management.create_process_paths import get_processing_list
import yaml
import glob

bin_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bins")

# ------------------------------------------------------------------------------
# Set default verbosity

if os.environ.get('RADICAL_ENTK_VERBOSE') == None:
    os.environ['RADICAL_ENTK_REPORT'] = 'True'

os.environ['RADICAL_VERBOSE'] = 'DEBUG'
os.environ['RADICAL_ENTK_VERBOSE'] = 'DEBUG'

# Description of how the RabbitMQ process is accessible
# No need to change/set any variables if you installed RabbitMQ has a system
# process. If you are running RabbitMQ under a docker container or another
# VM, set "RMQ_HOSTNAME" and "RMQ_PORT" in the session where you are running
# this script.
hostname = os.environ.get('RMQ_HOSTNAME', 'localhost')
port = int(os.environ.get('RMQ_PORT', 5672))

def read_yaml_file(filename):
    """read yaml file"""
    with open(filename) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def create_entry(cmt_filename, param_path, pipelinedir):
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

    # Earthquake file in the database
    cmt_file_db = os.path.join(eq_dir, "eq_" + eq_id + ".cmt")


    # Path to function
    create_database_func = os.path.join(bin_path, "create_entry.py")

    # Create a Stage object
    database_entry = Stage()

    t1 = Task()
    t1.name = 'database-entry'
    t1.pre_exec = [  # Conda activate
        DB_params["conda-activate"]]
    t1.executable = [DB_params['bin-python']]  # Assign executable to the task
    t1.arguments = [create_database_func, os.path.abspath(cmt_filename)]

    # In the future maybe to database dir as a total log?
    t1.stdout = os.path.join(pipelinedir, "database-entry." + eq_id + ".stdout")
    t1.stderr = os.path.join(pipelinedir, "database-entry." + eq_id + ".stderr")

    # Add Task to the Stage
    database_entry.add_tasks(t1)

    return database_entry

def data_request(cmt_file_db, param_path, pipelinedir):
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

    # # Path to function
    request_data_func = os.path.join(bin_path, "request_data.py")

    # Create a Stage object
    datarequest = Stage()

    datarequest_t = Task()
    datarequest_t.name = 'data-request'
    datarequest_t.pre_exec = [  # Conda activate
        DB_params["conda-activate"]]
    datarequest_t.executable = [DB_params['bin-python']]  # Assign executable
    # to the task
    datarequest_t.arguments = [request_data_func, cmt_file_db]

    # In the future maybe to database dir as a total log?
    datarequest_t.stdout = os.path.join(pipelinedir,
                                        "datarequest." + cmt_file_db[3:-4] +
                                        ".stdout")
    datarequest_t.stderr = os.path.join(pipelinedir,
                                        "datarequest." + cmt_file_db[3:-4] +
                                        ".stderr")

    # Add Task to the Stage
    datarequest.add_tasks(datarequest_t)

    return datarequest


def write_sources(cmt_file_db, param_path, pipelinedir):
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

    # Path to function
    write_source_func = os.path.join(bin_path, "write_sources.py")

    # Create a Stage object
    w_sources = Stage()
    w_sources.name = 'Write-Sources'

    # Create Task for stage
    w_sources_t = Task()
    w_sources_t.name = 'Task-Sources'
    w_sources_t.pre_exec = [  # Conda activate
        DB_params["conda-activate"]]
    w_sources_t.executable = [DB_params['bin-python']]  # Assign executable
    # to the task
    w_sources_t.arguments = [write_source_func, cmt_file_db]

    # In the future maybe to database dir as a total log?
    w_sources_t.stdout = os.path.join(pipelinedir,
                                      "write_sources." + cmt_file_db[3:-4] +
                                      ".stdout")
    w_sources_t.stderr = os.path.join(pipelinedir,
                                      "write_sources." + cmt_file_db[3:-4] +
                                      ".stderr")

    # Add Task to the Stage
    w_sources.add_tasks(w_sources_t)

    return w_sources


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
    runSF3d.name = 'Simulation'

    for at in attr[0]:
        sf_t = Task()
        sf_t.name = 'run-' + at

        # Module Loading
        sf_t.pre_exec = [  # Get rid of existing modules
            'module purge']
        for module in cm_dict["modulelist"]:
            sf_t.pre_exec.append("module load %s" % module)
        sf_t.pre_exec.append("module load %s" % cm_dict["gpu_module"])

        # Change directory to specfem directories
        sf_t.pre_exec.append(  # Change directory
            "cd %s" % os.path.join(simdir, at))

        sf_t.executable = ['./bin/xspecfem3D']  # Assign executable

        # In the future maybe to database dir as a total log?
        sf_t.stdout = os.path.join(pipelinedir,
                                   "run_specfem." + cmt_file_db[3:-4] +
                                   ".stdout")
        sf_t.stderr = os.path.join(pipelinedir,
                                   "run_specfem." + cmt_file_db[3:-4] +
                                   ".stderr")

        sf_t.gpu_reqs = {
            'processes': 6,
            'process_type': 'MPI',
            'threads_per_process': 1,
            'thread_type': None
        }

        sf_t.cpu_reqs = {
            'processes': 6,
            'process_type': 'MPI',
            'threads_per_process': 1,
            'thread_type': None
        }

        # Add Task to the Stage
        runSF3d.add_tasks(sf_t)

        return runSF3d


def specfem_clean_up(cmt_file_db, param_path, pipelinedir):
    """ Cleaning up the simulation directories since we don't need all the
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
    clean_up.name = 'Clean-Up'

    # Create Task for stage
    clean_up_t = Task()
    clean_up_t.name = 'Task-Clean-Up'
    clean_up_t.pre_exec = [  # Conda activate
        DB_params["conda-activate"]]
    clean_up_t.executable = [DB_params['bin-python']]  # Assign executable
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


def convert_traces(cmt_file_db, param_path):
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

    # File and directory
    cmt_dir = os.path.dirname(cmt_file_db)
    sim_dir = os.path.join(cmt_dir, "CMT_SIMs")


    ## Create a Stage object
    conversion_stage = Stage()
    conversion_stage.name = 'Conversion'

    # Conversion binary
    conversion_bin = os.path.join(bin_path, "convert_to_asdf.py")


    attr = ["CMT", "CMT_rr", "CMT_tt", "CMT_pp", "CMT_rt", "CMT_rp",
            "CMT_tp", "CMT_depth", "CMT_lat", "CMT_lon"]

    ##### Converting the synthetic data
    if DB_params['verbose']:
        print("\nConverting synthetic traces to ASDF ... \n")

    for _i, at in enumerate(attr[:DB_params["npar"] + 1]):

        # Path file
        syn_path_file = os.path.join(sim_dir, at, at + ".yml")

        # Create Task for stage
        c_task = Task()
        c_task.name = at

        c_task.pre_exec = [DB_params["conda-activate"]]
        c_task.executable = [DB_params['bin-python']]  # Assign executable
                                                       # to the task

        c_task.arguments = [conversion_bin,
                            "-f", syn_path_file,  # Path File
                            "-v", DB_params["verbose"],  # verbose flag
                            "-s", DB_params["verbose"],  # status bar
                            ]

        conversion_stage.add_tasks(c_task)


    ##### Converting the observed data
    if DB_params['verbose']:
        print("\nConverting observed traces to ASDF ... \n")

    obs_path_file = os.path.join(cmt_dir, "seismograms", "obs", "observed.yml")

    # Create Task for stage
    c_task = Task()
    c_task.name = at

    c_task.pre_exec = [DB_params["conda-activate"]]
    c_task.executable = [DB_params['bin-python']]  # Assign executable
    # to the task

    c_task.arguments = [conversion_bin,
                        "-f", obs_path_file,  # Path File
                        "-v", DB_params["verbose"],  # verbose flag
                        "-s", DB_params["verbose"],  # status bar
                        ]

    conversion_stage.add_tasks(c_task)

    return conversion_stage


def create_process_path_files(cmt_file_db, param_path, pipelinedir):
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

    # Process path function
    create_process_path_bin = os.path.join(pipelinedir,
                                           "create_path_files.py")

    # Create Process Paths Stage (CPP)
    # Create a Stage object
    cpp = Stage()
    cpp.name = 'CreateProcessPaths'

    # Create Task
    cpp_t = Task()
    cpp_t.name = "CPP-Task"
    cpp_t.pre_exec = [  # Conda activate
                      DB_params["conda-activate"]]
    cpp_t.executable = [DB_params['bin-python']]  # Assign executable
                                                  # to the task
    cpp_t.arguments = [create_process_path_bin, cmt_file_db]

    cpp.add_tasks(cpp_t)

    return cpp


def create_processing_stage(cmt_file_db, param_path):
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

    # Processing param dir
    process_obs_param_dir = os.path.join(param_path, "ProcessObserved")
    process_syn_param_dir = os.path.join(param_path, "ProcessSynthetic")

    # Process path list
    # Important step! This creates a processing list prior to having created
    # the actual process path files. It is tested so it definitely works!
    # this way the processes can be distributed for each ASDF file on one
    # processor or more (MPI enabled!)
    processing_list = get_processing_list(cmt_file_db, process_obs_param_dir,
                                          process_syn_param_dir, verbose=True)

    # Process path function
    process_func = os.join.path(bin_path, "process_asdf.py")

    # Create Process Paths Stage (CPP)
    # Create a Stage object
    process_stage = Stage()
    process_stage.name = 'Processing'

    # Loop over process path files
    for process_path in processing_list:

        # Create Task
        processing_task = Task()

        # This way the task gets the name of the path file
        processing_task.name = "Processing-" + os.path.basename(process_path)

        processing_task.pre_exec = [  # Conda activate
                                      DB_params["conda-activate"]]

        processing_task.executable = [DB_params['bin-python']]  # Assign exec.
                                                                # to the task

        processing_task.arguments = [process_func,
                                     "-f", process_path,
                                     "-v", DB_params["verbose"]]

        process_stage.add_tasks(processing_task)

    return process_stage


def create_windowing_stage(cmt_file_db, param_path):
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

    # Windowing parameter file directory
    window_process_dir = os.path.join(param_path, "CreateWindows")

    # Window path list
    # Important step! This creates a windowing list prior to having created
    # the actual window path files. It is tested so it definitely works!
    # This way the windowing processes can be distributed for each ASDF file
    # pair on one processor (No MPI support!)

    window_path_list = get_windowing_list(cmt_file_db, window_process_dir,
                                          verbose=False)

    # Process path function
    window_func = os.join.path(bin_path, "window_selection_asdf.py")

    # Create Process Paths Stage (CPP)
    # Create a Stage object
    window_stage = Stage()
    window_stage.name = 'Windowing'

    # Loop over process path files
    for window_path in window_path_list:

        # Create Task
        window_task = Task()

        # This way the task gets the name of the path file
        window_task.name = os.path.basename(window_path)

        window_task.pre_exec = [  # Conda activate
                                      DB_params["conda-activate"]]

        window_task.executable = [DB_params['bin-python']]  # Assign exec
                                                            # to the task

        window_task.arguments = [window_func,
                                 "-f", window_path,
                                 "-v", DB_params["verbose"]]

        window_stage.add_tasks(window_task)

    return window_stage


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
    inv_dict_stage.name = 'Creating'

    # Create Task
    inv_dict_task = Task()

    # This way the task gets the name of the path file
    inv_dict_task.name = "Inversion-Dictionaries"

    inv_dict_task.pre_exec = [  # Conda activate
                              DB_params["conda-activate"]]

    inv_dict_task.executable = [DB_params['bin-python']]  # Assign exec
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
    inversion_stage.name = 'CMT3D'

    # Create Task
    inversion_task = Task()

    # This way the task gets the name of the path file
    inversion_task.name = "Inversion"

    inversion_task.pre_exec = [  # Conda activate
        DB_params["conda-activate"]]

    inversion_task.executable = [DB_params['bin-python']]  # Assign exec
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

    # ---- Create Database Entry --------------------------------------------- #

    # Create Database entry stage:
    database_entry_stage = create_entry(cmt_filename, param_path, pipelinedir)

    # Add Stage to the Pipeline
    p.add_stages(database_entry_stage)

    # ---- REQUEST DATA ------------------------------------------------------ #

    # Request data stage
    datarequest_stage = data_request(cmt_file_db, param_path, pipelinedir)

    # Add Stage to the Pipeline
    p.add_stages(datarequest_stage)

    # ---- Write Sources ----------------------------------------------------- #

    # Create Source modification stage
    w_sources_stage = write_sources(cmt_file_db, param_path, pipelinedir)

    # Add Stage to the Pipeline
    p.add_stages(w_sources_stage)

    # ---- Run Specfem ------------------------------------------------------- #

    # Create Specfem Stage
    runSF3D_stage = run_specfem(cmt_file_db, param_path, pipelinedir)

    # Add Simulation stage to the Pipeline
    p.add_stages(runSF3D_stage)

    # ---- Clean Up Specfem -------------------------------------------------- #

    # Create clean_up stage
    clean_up_stage = specfem_clean_up(cmt_file_db, param_path, pipelinedir)

    # Add Stage to the Pipeline
    p.add_stages(clean_up_stage)

    # ---- Convert to ASDF --------------------------------------------------- #

    # Create conversion stage
    conversion_stage = convert_traces(cmt_file_db, param_path)

    # Add stage to pipeline
    p.add_stages(conversion_stage)

    # ---- Create Process Path files ----------------------------------------- #

    # Create Process Stage Pipeline
    process_path_stage = create_process_path_files(cmt_file_db,
                                                   param_path,
                                                   pipelinedir)
    p.add_stages(process_path_stage)

    # ---- Process Traces ---------------------------------------------------- #

    # Create processing stage
    processing_stage = create_processing_stage(cmt_file_db, param_path)

    p.add_stages(processing_stage)

    # ---- Window Traces ---------------------------------------------------- #

    # Create processing stage
    windowing_stage = create_processing_stage(cmt_file_db, param_path)

    p.add_stages(windowing_stage)

    # ============== RUNNING THE PIPELINE ==================================== #

    # Create Application Manager
    appman = AppManager(hostname=hostname, port=port)

    # Create a dictionary describe four mandatory keys:
    # resource, walltime, and cpus
    # resource is 'local.localhost' to execute locally
    res_dict = {
        'resource': 'princeton.tiger_gpu',
        'project': 'geo',
        'queue': 'gpu',
        'schema': 'local',
        'walltime': 300,
        'cpus': 6,
        'gpus': 6
    }

    # Assign resource request description to the Application Manager
    appman.resource_desc = res_dict

    # Assign the workflow as a set or list of Pipelines to the Application Manager
    # Note: The list order is not guaranteed to be preserved
    appman.workflow = set([p])

    # Run the Application Manager
    appman.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Path to CMTSOLUTION file',
                        type=str)
    args = parser.parse_args()

    # Run
    workflow(args.filename)
