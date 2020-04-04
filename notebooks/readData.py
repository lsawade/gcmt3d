import fnmatch
import os
import numpy as np
import csv


# from obspy.core import read

# %%
def getEventInfo():
    FirstLine = True;

    eventID = []
    datetime = []
    date = []
    time = []
    year = []
    month = []
    day = []

    with open('events', 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='|')

        for row in reader:
            if FirstLine:
                FirstLine = False
                continue
            eventID.append(row[0])

            tmp = row[1].split('T')
            date
            print(tmp)

            print(row[0])


def getStationInfo():
    FirstLine = True

    network = []
    station = []
    starttime = []
    endtime = []
    latitude = []
    longitude = []
    elevation = []

    with open('scsnstation', 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')

        for row in reader:
            if FirstLine:
                FirstLine = False
                continue
            network.append(row[0])
            station.append(row[1])
            starttime.append(row[2])
            endtime.append(row[3])
            latitude.append(row[4])
            longitude.append(row[5])
            elevation.append(row[6])

        latitude = [float(x) for x in latitude]
        longitude = [float(x) for x in longitude]
        elevation = [float(x) for x in elevation]

        # tmp = row[1].split('T')
        # date
        # print(tmp)

    return network, station, starttime, endtime, latitude, longitude, elevation


def writeStationInfo():
    '''
    Writing station file for SPECFEM INPUT
    '''

    network, station, starttime, endtime, latitude, longitude, elevation = \
        getStationInfo()

    file = open('STATIONS', 'w')

    for i, val in enumerate(network):
        file.write('{0:11s}{1:4s}{2:12.4f}{3:12.4f}{4:10.1f}{5:8.1f}\n'
                   .format(station[i], network[i], latitude[i], longitude[i],
                           elevation[i], 0))
        # burial
    file.close()


def readtrace(DIR, network, station, comp, ext):
    '''
    goes through directory finds tracefile
    DIR is the directory to look for the files
    '''

    string = network + '.' + station + '*' + comp + '*' + ext
    for file in os.listdir(DIR):
        if fnmatch.fnmatch(file, string):
            seisfile = os.path.join(DIR, file)
            break

    if ext == 'ascii':
        sc = np.loadtxt(seisfile)

    return sc[:, 0], sc[:, 1]


def collect_trace(lat, lon, t, a, width, height, m):
    '''
    Creates LineCollection object that is plottable on a map
    lat,lon ar origin
    t is time vector
    a is amplitude values
    width is the width in degrees of the seismogram
    height is the height in degrees of the seismogram
    m is map object
    '''

    t_scale = 1 / (np.max(t) - np.min(t)) * width
    a_scale = 1 / (np.max(a) - np.min(a)) * height

    # This calculates the seismogram in terms of lat and long
    scx, scy = m(lon + (t - t[0]) * t_scale, lat + a * a_scale)

    # This calculates the x axis location for the seismogram
    _, scy_corr = m(lon + (t - t[0]) * t_scale, lat + np.zeros([len(a)]))

    # This calculates the y axis axis location for the seismogram
    scx_corr, _ = m(lon + np.zeros([len(a)]), lat + a * a_scale)

    # This corrects the seismogram so that it's horizontal on the map
    scx = scx - (scx_corr - scx_corr[0])
    scy = scy - (scy_corr - scy_corr[0])

    return scx, scy


def readCMTsolution(CMT):
    '''
    This reads a CMT Solution text file and returns dictionary with it's
    values as in the CMT solution file just without spaces
    '''
    cmtFile = open(CMT, 'r')

    first_line = 0

    # create empty dictionary
    MT_dict = {}

    for line in cmtFile:
        # skip first line
        if first_line == 0:
            first_line = 1
            continue
        elif first_line == 1:
            # Splitting the line such the string and value are detach
            line = line.split(':')

            # adding to dictionary
            MT_dict[line[0].replace(" ", "")] = line[1].strip()

            first_line = 2
            continue

        # Splitting the line such the string and value are detach
        line = line.split(':')

        # adding to dictionary
        MT_dict[line[0].replace(" ", "")] = eval(line[1])

    return MT_dict
