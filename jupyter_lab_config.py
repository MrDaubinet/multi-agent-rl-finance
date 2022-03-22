import os
c = get_config()
os.environ['LD_PRELOAD'] = '/usr/local/lib/python3.8/dist-packages/torch/lib/libgomp-d22c30c5.so.1'
c.Spawner.env.update('LD_PRELOAD')