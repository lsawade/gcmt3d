from radical.entk import Pipeline, Stage, Task, AppManager
import os
import argparse
from get_eq_dir import get_eq_entry_path
import yaml

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

def main(cmt_filename):
    '''This tiny function runs shit

    Args:
        cmt_filename: str containing the path to the cmt solution that is
                      supposed to be inverted for

    Usage:
        From the commandline:
            python pipeline <path/to/cmtsolution>

    '''

    # Path to pipeline file
    pipelinepath = os.path.abspath(__file__)
    pipelinedir = os.path.dirname(pipelinepath)

    # Define parameter directory
    param_path = os.path.join(os.path.dirname(pipelinedir), "params")
    databaseparam_path = os.path.join(param_path,
                                      "Database/DatabaseParameters.yml")
    DB_params = read_yaml_file(databaseparam_path)
    print(DB_params)

    # Earthquake specific database parameters
    # Dir and eq_id
    eq_dir, eq_id = get_eq_entry_path(DB_params["databasedir"], cmt_filename)
    # Earthquake file in the database
    cmt_file_db = os.path.join(eq_dir, "eq_" + eq_id + ".cmt")

    # Create a Pipeline object
    p = Pipeline()

    # ---- DATABASE ENTRY TASK ---------------------------------------------- #

    # Path to function
    create_database_func = os.path.join(pipelinedir,
                                        "01_Create_Database_Entry.py")

    # Create a Stage object
    database_entry = Stage()

    t1 = Task()
    t1.name = 'database-entry'
    t1.pre_exec = [# Conda activate
                   DB_params["conda-activate"]]
    t1.executable = [DB_params['bin-python']]  # Assign executable to the task
    t1.arguments = [create_database_func, os.path.abspath(cmt_filename)]

    # In the future maybe to database dir as a total log?
    t1.stdout = os.path.join(pipelinedir, "database-entry." + eq_id + ".stdout")
    t1.stderr = os.path.join(pipelinedir, "database-entry." + eq_id + ".stderr")

    # Add Task to the Stage
    database_entry.add_tasks(t1)

    # Add Stage to the Pipeline
    p.add_stages(database_entry)

    # # ---- REQUEST DATA ----------------------------------------------------- #
    #
    # # Path to function
    # request_data_func = os.path.join(pipelinedir, "02_Request_Data.py")
    #
    # # Create a Stage object
    # datarequest = Stage()
    #
    # datarequest_t = Task()
    # datarequest_t.name = 'data-request'
    # datarequest_t.pre_exec = [  # Conda activate
    #     DB_params["conda-activate"]]
    # datarequest_t.executable = [DB_params['bin-python']]  # Assign executable
    #                                                       # to the task
    # datarequest_t.arguments = [request_data_func, cmt_file_db]
    #
    # # In the future maybe to database dir as a total log?
    # datarequest_t.stdout = os.path.join(pipelinedir,
    #                                   "datarequest." + eq_id + ".stdout")
    # datarequest_t.stderr = os.path.join(pipelinedir,
    #                                   "datarequest." + eq_id + ".stderr")
    #
    # # Add Task to the Stage
    # datarequest.add_tasks(datarequest_t)
    #
    # # Add Stage to the Pipeline
    # p.add_stages(datarequest)

    # ---- Write Sources ---------------------------------------------------- #

    # Path to function
    write_source_func = os.path.join(pipelinedir, "03_Write_Sources.py")

    # Create a Stage object
    w_sources = Stage()
    w_sources.name = 'Write-Sources'

    # Create Task for stage
    w_sources_t = Task()
    w_sources_t.name = 'Write-Sources'
    w_sources_t.pre_exec = [  # Conda activate
        DB_params["conda-activate"]]
    w_sources_t.executable = [DB_params['bin-python']]  # Assign executable
    # to the task
    w_sources_t.arguments = [write_source_func, cmt_file_db]

    # In the future maybe to database dir as a total log?
    w_sources_t.stdout = os.path.join(pipelinedir,
                                        "write_sources." + eq_id + ".stdout")
    w_sources_t.stderr = os.path.join(pipelinedir,
                                        "write_sources." + eq_id + ".stderr")

    # Add Task to the Stage
    w_sources.add_tasks(w_sources_t)

    # Add Stage to the Pipeline
    p.add_stages(w_sources)

    # ---- Run Specfem ----------------------------------------------------- #

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

    simdir = os.path.join(eq_dir, "CMT_SIMs")

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
                                      "run_specfem." + eq_id + ".stdout")
        sf_t.stderr = os.path.join(pipelinedir,
                                      "run_specfem." + eq_id + ".stderr")

        sf_t.gpu_reqs = {
            'processes': 6,
            'process_type': 'MPI',
            'threads_per_process': 1,
            'thread_type': 'OpenMP'
            }

        # Add Task to the Stage
        runSF3d.add_tasks(sf_t)

    # Add Simulation stage to the Pipeline
    p.add_stages(runSF3d)


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
        'cpus': 2,
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
    main(args.filename)
