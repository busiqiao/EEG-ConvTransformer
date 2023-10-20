import glob
import os
import platform
from tqdm import tqdm
from data_load.read_mat import read_eeg_mat, read_locs_xlsx
from joblib import Parallel, delayed
import pickle
import numpy as np
from preprocessing.aep import azim_proj, gen_images
import einops

parallel_jobs = 1
locs = None


def thread_read_write(x, y, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time, channels=124], y
    """
    os.makedirs(os.path.dirname(pkl_filename), exist_ok=True)
    with open(pkl_filename + '.pkl', 'wb') as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def go_through(filenames, pkl_path):
    for f in tqdm(filenames, desc=' Total', position=0, leave=True, colour='YELLOW', ncols=80):
        eeg, y = read_eeg_mat(f)  # [n_samples=5184, t_length=32, channels=124]

        name = f.split('/')[-1].replace('.mat', '')
        sub_pkl_path = pkl_path + name

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

        Parallel(n_jobs=parallel_jobs)(
            delayed(thread_read_write)(imgs[i], y[i], sub_pkl_path + name + '_' + str(i) + '_' + str(y[i]))
            for i in tqdm(range(len(y)), desc=' write ' + name, position=1, leave=False, colour='WHITE', ncols=80))


def file_scanf(path, endswith, sub_ratio=1):
    files = glob.glob(path + '/*')
    if platform.system().lower() == 'windows':
        files = [f.replace('\\', '/') for f in files]
    disallowed_file_endings = (".gitignore", ".DS_Store")
    _input_files = files[:int(len(files) * sub_ratio)]
    return list(filter(lambda x: not x.endswith(disallowed_file_endings) and x.endswith(endswith), _input_files))


if __name__ == "__main__":
    path = 'H:/EEG/EEGDATA'
    filenames = file_scanf(path, endswith='.mat')
    locs = read_locs_xlsx('H:/EEG/EEG-ConvTransformer/data_load/electrodes_locations/GSN_HydroCel_128.xlsx')

    go_through(filenames, pkl_path=path + '/img_pkl_124/')
