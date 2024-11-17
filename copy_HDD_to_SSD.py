import os
import tqdm
import shutil
import pickle

reload_hhd_data = False

HHD_path = '/Volumes/old_disk/match_image'
SSD_path = '/Volumes/Capstone'
# HHD_path = '/Users/olivereielson/Desktop/test'
# SSD_path = '/Users/olivereielson/Desktop/test2'


log_file = 'log.txt'

# Check if paths exist
assert os.path.exists(HHD_path), 'HHD path does not exist'
assert os.path.exists(SSD_path), 'SSD path does not exist'
assert os.path.exists(log_file), 'Log file does not exist'

# load in files that were already copied
with open(log_file, 'r') as f:
    copied_files = f.readlines()
    copied_files = [file_name.strip() for file_name in copied_files]


print(f'Number of files already copied: {len(copied_files)}')
#load in all files on HDD

if reload_hhd_data:
    print("Loading all files on HDD...This make take a while")
    all_files = os.listdir(HHD_path)
    print("overwriting pickle file")

    pickle_file_path = "File_names.pkl"
    with open(pickle_file_path, 'wb') as pkl_file:
        pickle.dump(all_files, pkl_file)
else:
    print("Loading all files on HDD from pickle file")
    all_files = pickle.load(open("File_names.pkl", "rb"))




# copy files to SSD
for file in tqdm.tqdm(all_files):



    if not file.endswith('.jpg') and not file.endswith('.xml'):
        continue

    # check if file was already copied
    if file in copied_files:
        # print(f'File {file} was already copied')
        continue
    # print(file)
    date = file.split('--')[1][:10]
    if not os.path.exists(f'{SSD_path}/{date}'):
        os.makedirs(f'{SSD_path}/{date}')


    shutil.copy(f'{HHD_path}/{file}', f'{SSD_path}/{date}/{file}')
    assert os.path.exists(f'{SSD_path}/{date}/{file}'), f'File {file} was not copied to SSD'
    with open(log_file, 'a') as f:
        f.write(f'{file}\n')