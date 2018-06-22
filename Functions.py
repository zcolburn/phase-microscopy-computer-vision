'''
Functions used to process the data used to train the model specified in
Main.py.
'''

###############################################################################
# Imports
import pandas as pd
import numpy as np
from skimage import io
import re
from glob import glob
import os
import shutil


###############################################################################
# Parameters
# Image dimensions.
xdim = 1392
ydim = 1040

# Ignore pixels within this distance from the image edge...
bumper = 100

# Create boxes with the below radius that are spaced every box_sep.
box_rad = 30
box_sep = 40


###############################################################################
# Define functions.


def map_Cache():
    """
    This function returns a data frame describing the Cache directory files.
    :param directory: The directory of interest.
    :return: A data frame describing the files and their relationships.
    """

    directory = 'Cache'

    # Determine paths to all files and folders.
    paths = glob(directory + '/**', recursive=True)

    # Retain only paths that point to files.
    paths = [i for i in paths if bool(re.search('\.', i))]

    # Put the data into a data frame.
    FileMap = pd.DataFrame(paths, columns=['path'])
    FileMap['FileType'] = [re.sub('^.*\.','',i) for i in paths]
    FileMap['Terminus'] = [re.sub('^.*\\\\','',i) for i in paths]
    FileMap['Terminus'] = [re.sub('.npy','',i) for i in FileMap['Terminus']]

    ends = FileMap['Terminus']
    matches = [re.match(r"([a-z]+)([0-9]+)", i, re.I).groups() for i in ends]
    FileMap['FileClass'] = [a for a, b in matches]
    FileMap['FileNumber'] = np.array([b for a, b in matches])

    return FileMap


def arrange_data(small_img,
                 outline,
                 box_sep=box_sep,
                 box_rad=box_rad,
                 bumper=bumper):
    """
    Create small partitions in the image for training purposes.
    :param small_img: A numpy array of dimensions (Rows, Cols).
    :param outline: A pd object of outline coordinates.
    :param box_sep: The separation between template box centers.
    :param box_rad: The radius of template boxes.
    :return: Numpy arrays X and Y for training. The coordinates these points
    correspond to.
    """
    y_dim, x_dim = small_img.shape

    y_pegs = np.array(range(0, int(y_dim/box_sep)-2))*box_sep+box_rad
    y_pegs = y_pegs[(y_pegs > bumper) & (y_dim - y_pegs > bumper)]
    y_pegs = list(y_pegs)
    x_pegs = np.array(range(0, int(x_dim/box_sep)-2))*box_sep+box_rad
    x_pegs = x_pegs[(x_pegs > bumper) & (x_dim - x_pegs > bumper)]
    x_pegs = list(x_pegs)

    rows = len(y_pegs)*len(x_pegs)
    dim = 2*box_rad+1
    cols = dim**2

    X = np.zeros((rows, dim, dim), dtype=int)
    Y = np.zeros(rows)

    Y_img = np.zeros((y_dim, x_dim))
    for row in np.array(outline):
        Y_img[row[1]-1, row[0]-1] = 1

    Coords = pd.DataFrame({'Row':np.zeros(rows),'Col':np.zeros(rows)})

    id = 0
    for row in y_pegs:
        row_min = row - box_rad
        row_max = row + box_rad + 1
        group = 0
        for col in x_pegs:
            col_min = col - box_rad
            col_max = col + box_rad + 1
            X[id,:,:] = small_img[row_min:row_max,col_min:col_max]
            box_sum = Y_img[row_min:row_max,col_min:col_max].sum()
            if (box_sum == 0) and (group == 0):
                Y[id]=group
            elif  (box_sum > 0) and ((group == 0) or (group == 1)):
                group=1
                Y[id]=group
            else:
                group=2
                Y[id]=group
            Coords['Row'][id] = row
            Coords['Col'][id] = col
            id += 1

    return (X, Y, Coords)


def create_training_data():
    FileMap = load_metadata()

    FOVGroups = list(set(FileMap['FOVGroup'][:].tolist()))

    iteration = 0
    max_iteration = sum(FileMap['FileType'] == 'txt')
    for FOVGroup in FOVGroups:
        fov_df = FileMap[np.array(FileMap.FOVGroup == str(FOVGroup))]
        img_df = fov_df[(fov_df.FileType == 'tif')]
        outline_df = fov_df[(fov_df.FileType == 'txt')]

        fovGroup = fov_df['FOVGroup'].iloc[0]

        # Load tiff stack.
        img = load_images(img_df['path'].iloc[0])

        # Iterate over frames.
        for i in range(0, len(outline_df)):
            frame = outline_df['Frame'].iloc[i]
            if np.isnan(frame):
                continue
            frame = int(frame)
            outline_id = int(outline_df['FileID'].iloc[i])
            small_img, outline = load_data(img,
                                           fovGroup,
                                           frame,
                                           outline_df,
                                           xdim=xdim,
                                           ydim=ydim,
                                           bumper=bumper)

            X, Y, Coords = arrange_data(small_img,
                                        outline,
                                        box_sep=box_sep,
                                        box_rad=box_rad,
                                        bumper=bumper)
            save_X(X, outline_id)
            save_Y(Y, outline_id)
            save_Coords(Coords, outline_id)

            iteration += 1
            print(str(iteration)+' of '+str(max_iteration)+' files.')



    print("All done!")


def load_outline(filename, xdim=xdim, ydim=ydim, bumper=bumper):
    """
    Load an outline txt file.
    :param filename: The name of the txt file.
    :param xdim: The width of the image in pixels.
    :param ydim: The height of the image in pixels.
    :param bumper: The buffer around the image with which to exclude the edge
    boundary.
    :return: A DataFrame containing the outline coordinates
    """
    outline = pd.read_csv(filename, sep='\\t', header=None)
    outline = outline[np.array((outline[0] > bumper) &
                               (outline[1] > bumper) &
                               (outline[0] - xdim < -bumper) &
                               (outline[1] - ydim < -bumper))]
    outline.rename(columns={0: 'Col', 1: 'Row'}, inplace=True)

    return outline


def load_images(filename):
    """
    Load a tif.
    :param filename: The name of the tif file.
    :return: A numpy array of shape (frames, Rows, Cols).
    """

    img = io.imread(filename)

    return img


def load_data(img,
              fovGroup,
              frame,
              FileMap,
              xdim=xdim,
              ydim=ydim,
              bumper=bumper):
    """
    Return a single image slice and the outline that corresponds to it.
    :param img: The image stack loaded from load_images.
    :param fovGroup: The FOVGroup of interest.
    :param frame: The frame of interest.
    :param df: The FileMap DataFrame filtered to include only the row
    corresponding to the frame of interest
    :param xdim: The number of pixels in the x direction.
    :param ydim: The number of pixels in the y direction.
    :param bumper: The buffer around the image with which to exclude the edge
    boundary.
    :return: A numpy array of dimensions (Rows, Cols). A pd object of outline
    coordinates.
    """
    df = FileMap.loc[
        (FileMap['Frame'] == frame) & (FileMap['FOVGroup'] == fovGroup)]
    small_img = img[frame,:,:]
    outline = load_outline(
        df['path'].iloc[0], xdim=xdim, ydim=ydim, bumper=bumper)
    return (small_img, outline)


def load_metadata():
    '''
    Load FileMap from the MetaData folder.
    :return: None
    '''
    input = pd.read_pickle('MetaData/FileMap.pkl')
    return input


def load_X(id):
    '''
    Load an X file given a file id number.
    :param id: File id number.
    :return: X
    '''
    in_file = 'Cache/X'+str(id)+'.npy'
    input = np.load(in_file)
    return input


def load_Y(id):
    '''
    Load a Y file given a file id number.
    :param id: File id number.
    :return: Y
    '''
    in_file = 'Cache/Y' + str(id)+'.npy'
    input = np.load(in_file)
    return input


def load_Coords(id):
    '''
    Load a Coords file given a file id number.
    :param id: File id number.
    :return: Coords
    '''
    in_file = 'Cache/Coords' + str(id)+'.npy'
    input = np.load(in_file)
    return input


def map_files(directory, frames_per_interval):
    """
    This file takes an input directory and returns a data frame describing
    the files contained within.
    :param directory: The directory of interest.
    :return: A data frame describing the files and their relationships.
    """

    # Determine paths to all files and folders.
    paths = glob(directory + '/**', recursive=True)

    # Retain only paths that point to files.
    paths = [i for i in paths if bool(re.search('\.', i))]

    # Put the data into a data frame.
    FileMap = pd.DataFrame(paths, columns=['path'])
    FileMap['FileType'] = [re.sub('^.*\.','',i) for i in paths]

    def get_minutes(X):
        """
        Determine the number of minutes the outline file refers to.
        :param X: The filename string.
        :return: A number or None.
        """
        try:
            return int(X.partition('m.txt')[0].rpartition('_')[2])
        except Exception as e:
            return None

    FileMap['Time'] = [get_minutes(i) for i in paths]

    FileMap['FOVGroup'] = [i.rpartition('\\')[0] for i in paths]
    fov_grp = FileMap['FOVGroup']
    FileMap['WellGroup'] = [i.rpartition('\\')[0] for i in fov_grp]
    well_grp = FileMap['WellGroup']
    FileMap['Condition'] = [i.rpartition('\\')[0] for i in well_grp]
    FileMap['Experiment'] = [i.rpartition('\\')[0] for i in FileMap[
        'Condition']]
    FileMap['Condition'] = [i.rpartition('\\')[2] for i in FileMap[
        'Condition']]
    FileMap['Experiment'] = [i.rpartition('\\')[2] for i in FileMap[
        'Experiment']]

    interval = np.min(FileMap['Time'][FileMap['Time'] > 0])

    FileMap['Frame'] = FileMap['Time']/(interval*frames_per_interval)

    FileMap['FileID'] = list(range(0, len(FileMap['Frame'])))

    return FileMap


def save_file_map(FileMap):
    '''
    Save the FileMap object.
    :param FileMap: A DataFrame describing input files.
    :return: None.
    '''
    pd.to_pickle(FileMap, 'MetaData/FileMap.pkl')


def clear_metadata_folder():
    '''
    Clear the contents of the MetaData folder.
    :return: None.
    '''
    if os.path.exists('MetaData'):
        shutil.rmtree('MetaData')
    os.makedirs('MetaData')


def save_X(X, id):
    '''
    Save an X object given X and a file id number.
    :param X: The image as a numpy array.
    :param id: File id number.
    :return: None.
    '''
    out_file = 'Cache/X' + str(id)
    np.save(out_file, X)


def save_Y(Y, id):
    '''
    Save a Y object given Y and a file id number.
    :param Y: A numpy array given the classification of the corresponding
    index in X.
    :param id: File id number.
    :return: None.
    '''
    out_file = 'Cache/Y' + str(id)
    np.save(out_file, Y)


def save_Coords(Coords, id):
    '''
    Save a Coords object given Coords and a file id number.
    :param Coords: The image as a numpy array.
    :param id: File id number.
    :return: None.
    '''
    out_file = 'Cache/Coords' + str(id)
    np.save(out_file, Coords)


def clear_cache_folder():
    '''
    Clear the contents of the Cache folder.
    :return: None.
    '''
    if os.path.exists('Cache'):
        shutil.rmtree('Cache')
    os.makedirs('Cache')
