from pathlib import Path

from data import train_generator, gen_train

DATA_AUGMENTATION = False
pth = Path('data')

if __name__ == '__main__':

    # setup data augmentation
    data_gen_args = dict()
    if DATA_AUGMENTATION:
        data_gen_args = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True,
                             fill_mode='nearest')

    # initialize generator of training data
    my_generator = train_generator(
        batch_size=20,
        train_path=pth / 'membrane/train',
        image_folder='image',
        mask_folder='label',
        aug_dict=data_gen_args,
        save_to_dir=pth / 'membrane/train/aug',
    )

    # you will see 60 transformed images and their masks in data/membrane/train/aug
    num_batch = 3
    for i, batch in enumerate(my_generator):
        if i >= num_batch:
            break

    ## create .npy data

    image_arr, mask_arr = gen_train(
        pth / 'membrane/train/aug/',
        pth / '/membrane/train/aug/',
    )

    # import numpy as np
    # np.save(pth / 'image_arr.npy', image_arr)
    # np.save(pth / 'mask_arr.npy', mask_arr)
