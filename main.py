# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path

from keras.callbacks import ModelCheckpoint

from data import train_generator, test_generator, save_result
from model import unet

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
    pth = Path('/content/bio_segmentation_keras/data')
else:
    pth = Path('data')
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = train_generator(
    batch_size=2,
    train_path=pth / 'membrane/train',
    image_folder='image',
    mask_folder='label',
    aug_dict=data_gen_args,
    save_to_dir=None
)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5',
                                   monitor='loss',
                                   verbose=1,
                                   save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

testGene = test_generator(test_path=pth / 'membrane/test')
results = model.predict_generator(testGene, 30, verbose=1)
save_result(pth / 'membrane/test', results)
