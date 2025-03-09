import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU configurada correctamente")
    except RuntimeError as e:
        print(e)
else:
    print("❌ No se detectó GPU")

print(tf.sysconfig.get_build_info())

