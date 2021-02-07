import os
import numpy as np
from ...utils.io import read_yaml_file
from ...source import CMTSource


def create_full_tensor(atensor):
    return np.array([[atensor[0], atensor[3], atensor[4]],
                     [atensor[3], atensor[1], atensor[5]],
                     [atensor[4], atensor[5], atensor[2]]])


def compute_angle(tensor1, tensor2):
    return (np.arccos(np.tensordot(tensor1, tensor2)
                      / np.sqrt(np.tensordot(tensor1, tensor1))
                      / np.sqrt(np.tensordot(tensor2, tensor2))))/np.pi*180.0


def check_change_in_params(cmtfile, new_cmtfile, paramdir) -> bool:
    """Checks whether the cmt has changed too much. If the change is ok,
    the function returns ``False`` if change is too large. Limits are taken
    from the Parameter files used for the inversion.

    Parameters
    ----------
    cmtfile : CMTSource
        starting solution
    new_cmtfile : CMTSource
        new solution
    paramdir : str
        parameter directory

    Returns
    -------
    bool
        returns ``False`` if change is too large
    """

    # Get CMT files
    cmtsource = CMTSource.from_CMTSOLUTION_file(cmtfile)
    new_cmtsource = CMTSource.from_CMTSOLUTION_file(new_cmtfile)

    # Load params
    paramsfile = os.path.join(paramdir, "Database", "DatabaseParameters.yml")
    params = read_yaml_file(paramsfile)
    outlier = params["outliers"]

    # Get tensors
    cmtt = cmtsource.tensor
    fcmtt = create_full_tensor(cmtt)
    ncmtt = new_cmtsource.tensor
    fncmtt = create_full_tensor(ncmtt)

    # Compute changes
    change_dict = dict(
        dM0=(new_cmtsource.M0 - cmtsource.M0)/cmtsource.M0,
        dtensor=np.max((ncmtt - cmtt)/cmtt),
        dangle=compute_angle(fncmtt, fcmtt),
        dlat=np.abs(new_cmtsource.latitude - cmtsource.latitude),
        dlon=np.abs(new_cmtsource.longitude - cmtsource.longitude),
        dz=np.abs(new_cmtsource.depth_in_m - cmtsource.depth_in_m)/1000.0
    )

    # Check
    checkdict = dict()

    for key, value in change_dict.items():
        checkdict[key] = value <= outlier[key]
        if checkdict[key] is False:
            print(f"{key} is larger than {outlier[key]}.")

    print(checkdict)
    print(change_dict)
    checklist = [x for x in checkdict.values()]

    return all(checklist)


def bin():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cmtfile", type=str,
                        help="cmtfilename")
    parser.add_argument("new_cmtfile", type=str,
                        help="new cmtfilename")
    parser.add_argument("paramdir", type=str,
                        help="parameter directory")

    args = parser.parse_args()

    print(check_change_in_params(args.cmtfile, args.new_cmtfile,
                                 args.paramdir))
