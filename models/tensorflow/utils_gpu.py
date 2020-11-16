import os
import tensorflow as tf

def set_session(**kwargs):
    if os.getenv('CUDA_VISIBLE_DEVICES') == -1:
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for gpu in kwargs.get("gpus", [0]):
            tf.config.experimental.set_virtual_device_configuration(physical_devices[gpu], 
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=kwargs.get("gpu_memory", 8096))])
            #tf.config.experimental.set_memory_growth(physical_devices[gpu], True)

        tf.config.experimental.set_visible_devices(devices= [physical_devices[id] for id in kwargs.get("gpus", [0])], device_type='GPU')

