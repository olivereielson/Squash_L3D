import os
import tqdm
import shutil

HHD_path = '/Users/olivereielson/Desktop/test'
SSD_path = '/Users/olivereielson/Desktop/test2'

log_file = 'log.txt'

# Check if paths exist
assert os.path.exists(HHD_path), 'HHD path does not exist'
assert os.path.exists(SSD_path), 'SSD path does not exist'
assert os.path.exists(log_file), 'Log file does not exist'

# load in files that were already copied
with open(log_file, 'r') as f:
    copied_files = f.readlines()


#load in all files on HDD
all_files = os.listdir(HHD_path)



# copy files to SSD
for file in tqdm.tqdm(all_files):

    # check if file was already copied
    if file in copied_files:
        continue

    date = file.split('--')[1][:10]
    if not os.path.exists(f'{SSD_path}/{date}'):
        os.makedirs(f'{SSD_path}/{date}')


    shutil.copy(f'{HHD_path}/{file}', f'{SSD_path}/{date}/{file}')
    assert os.path.exists(f'{SSD_path}/{date}/{file}'), f'File {file} was not copied to SSD'
    with open(log_file, 'a') as f:
        f.write(f'{file}\n')