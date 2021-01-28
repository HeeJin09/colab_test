from model import *
from data import *
import tensorflow as tf
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

data_gen_args = dict(rotation_range=0.25,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'datasets','input','label',data_gen_args,save_to_dir = None)

model = unet()
def train():
    start_time = time.time()
    model_checkpoint = ModelCheckpoint('covidtest.hdf5', monitor='loss',verbose=1, save_best_only=True)
    model.fit_generator(myGene,steps_per_epoch=300,epochs=50,callbacks=[model_checkpoint])
    print('total time(min):', (time.time()-start_time)/60)
def test():
    model.load_weights('/covidtest.hdf5')
    testGene = testmyGenerator("./drive/MyDrive/test/datasets/test_image")
    results = model.predict_generator(testGene,3,verbose=1)
    saveResult("./drive/MyDrive/test/datasets/result",results)

#train()
test()
