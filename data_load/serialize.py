import glob
import os
import platform
import multiprocessing
from data_load.read_mat import read_eeg_mat, read_locs_xlsx
import pickle
import numpy as np
from preprocessing.aep import azim_proj, gen_images
import einops


def save_data(args):
    filename, f, pkl_path, locs = args  # Add locs to the arguments
    print(filename + ' start')
    eeg, y1, y2 = read_eeg_mat(f)  # [n_samples=5184, t_length=32, channels=124]

    # -----------------
    samples, time, channels = np.shape(eeg)
    eeg = einops.rearrange(eeg, 'n t c -> (n t) c', n=samples, t=time, c=channels)

    locs_2d = [azim_proj(e) for e in locs]

    imgs = gen_images(locs=np.array(locs_2d),  # [samples*time, colors, W, H]
                      features=eeg,
                      n_gridpoints=32,
                      normalize=True).squeeze()

    imgs = einops.rearrange(imgs, '(n t) w h -> n t w h', n=samples, t=time)
    # -------------------

    data = []
    labels = []
    for i in range(len(y1)):
        data.append(imgs[i])
        labels.append((y1[i], y2[i]))

    # Save data and labels into one mat_file
    with open(pkl_path + f'{filename}.pkl', 'wb') as file:
        pickle.dump((data, labels), file)
        print(f'Saved {filename}.pkl')


def go_through(filenames, pkl_path, locs):  # Add locs to the arguments
    file_names = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(save_data, [(name, f, pkl_path, locs) for name, f in zip(file_names, filenames)])


def file_scanf(path, endswith, sub_ratio=1):
    files = glob.glob(path + '/*')
    if platform.system().lower() == 'windows':
        files = [f.replace('\\', '/') for f in files]
    disallowed_file_endings = (".gitignore", ".DS_Store")
    _input_files = files[:int(len(files) * sub_ratio)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(endswith), _input_files))


if __name__ == "__main__":
    data_path = '/data/EEG72'
    mat_file = file_scanf(data_path, endswith='.mat')
    loc = read_locs_xlsx('/home/zdd/Desktop/Projects/EEG-ConvTransformer/data_load/electrodes_locations/GSN_HydroCel_128.xlsx')

    go_through(mat_file, pkl_path='/data/EEG72-IMG/', locs=loc)
