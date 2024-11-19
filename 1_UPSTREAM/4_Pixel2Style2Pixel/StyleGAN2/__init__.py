import sys
import os

sys.path.append(os.path.abspath('/workspace/wjj910'))
sys.path.append(os.path.abspath('/workspace/wjj910/Task1_clf'))
sys.path.append(os.path.abspath('/workspace/wjj910/Task1_clf/Task1_inversion'))
sys.path.append(os.path.abspath('/workspace/wjj910/Task1_clf/Task1_inversion/StyleGAN2_3d_OWN'))
sys.path.append(os.path.abspath('/workspace/wjj910/Task1_clf/Task1_inversion/StyleGAN2_3d_OWN/stylegan2_pytorch'))

from stylegan2_pytorch import Trainer, StyleGAN2, NanException, ModelLoader
