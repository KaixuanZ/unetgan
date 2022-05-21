import sys
sys.path.insert(0, 'unetgan')
from inference import run
import json

with open('./unetgan/configs/unetgan.json') as f:
    config = json.load(f)
print(config)
# config["test_mode"] = True
config["test_mode"] = False
G_unetgan, D_unetgan, z_unetgan, y_unetgan = run(config)

import pdb;pdb.set_trace()

sys.path.insert(0, 'stylegan2')