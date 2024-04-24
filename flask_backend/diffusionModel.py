import os
import shutil
# libs
from dataclasses import dataclass
import datasets
from diffusers import UNet2DModel

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from inverse_pipeline.DDIM_Scheduler import DDIMScheduler
from inverse_pipeline.inverse_pipeline import Inverse_Pipeline
# import matplotlib.pyplot as plt

class TestModel:
    def __init__(self, root_path:str) -> None:
        self.root_path = root_path
        self.num_mask = 0
        self.num_final = 0

    def get_mask_id(self):
        return f"{self.num_mask}"

    def get_final_id(self):
        return f"{self.num_final}"

    def gen_mask(self, src_name:str, dst_name:str):
        src_path = os.path.join(self.root_path, src_name)
        dst_path = os.path.join(self.root_path, dst_name)
        shutil.copy(src_path, dst_path)
        self.num_mask += 1
        return f'{self.num_mask}'

    def gen_final(self, src_name:str, dst_name:str):
        src_path = os.path.join(self.root_path, src_name)
        dst_path = os.path.join(self.root_path, dst_name)
        shutil.copy(src_path, dst_path)
        self.num_final += 1
        return f'{self.num_final}'

@dataclass
class TestingConfig:
    num_train_timesteps=1000
    image_size = 32  # the generated image resolution
    test_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 2040
    gradient_accumulation_steps = 1
    learning_rate = 2e-5
    lr_warmup_steps = 5000
    save_image_epochs = 10
    save_model_epochs = 100
    num_inference_steps = 100
    eta=0.85
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-celeba"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

class DiffusionModel:
    
    def __init__(self, root_path:str) -> None:
        self.root_path = root_path
        self.num_mask = 0
        self.num_final = 0
        # model configs
        self.config = TestingConfig()
        self.eta = 0.85
        self.num_inference_steps = 100
        self.model = UNet2DModel.from_pretrained(self.config.output_dir + '/unet', use_safetensors=True)
        self.model = self.model.cuda()
        self.noise_scheduler = DDIMScheduler(config=self.config, num_train_timesteps=1000)
        self.pipeline = Inverse_Pipeline(unet=self.model, scheduler=self.noise_scheduler)
        # init color map
        self.ColorMap = {
            'background':   {'id': 0, 'color': '#000000'},
            'skin':         {'id': 1, 'color': '#cc0000'},
            'nose':         {'id': 2, 'color': '#4c9900'},
            'eye_glasses':  {'id': 3, 'color': '#cccc00'},
            'eye_left':     {'id': 4, 'color': '#3333ff'},
            'eye_right':    {'id': 5, 'color': '#cc00cc'},
            'brow_left':    {'id': 6, 'color': '#00ffff'},
            'brow_right':   {'id': 7, 'color': '#ffcccc'},
            'ear_left':     {'id': 8, 'color': '#663300'},
            'ear_right':    {'id': 9, 'color': '#ff0000'},
            'mouth':       {'id': 10, 'color': '#66cc00'},
            'lip_up':      {'id': 11, 'color': '#ffff00'},
            'lip_low':     {'id': 12, 'color': '#000099'},
            'hair':        {'id': 13, 'color': '#0000cc'},
            'hat':         {'id': 14, 'color': '#ff3399'},
            'ear_ring':    {'id': 15, 'color': '#00cccc'},
            'necklace':    {'id': 16, 'color': '#003300'},
            'neck':        {'id': 17, 'color': '#ff9933'},
            'cloth':       {'id': 18, 'color': '#00cc00'},
        }
    def get_mask_id(self):
        return f"{self.num_mask}"

    def get_final_id(self):
        return f"{self.num_final}"

    def inverse(self, clean_images, A, num_inference_steps=100, eta=0.85):
        output = self.pipeline(
            batch_size=1,
            y_0=clean_images,
            A=A,
            num_inference_steps=num_inference_steps,
            eta=eta
        )
        # out_array = output['model_output'].detach().cpu().numpy()
        # print(out_array.shape)
        return output['model_output'].detach().cpu().numpy()

    def numpy_to_image(self, np_array):
        image = Image.fromarray(np_array, 'RGB')
        return image

    def colorHexToRGB(self, colorHex:str):
        if colorHex[0] == '#' and len(colorHex)==7:
            HexCode = colorHex[1:]
            r = int(HexCode[0:2], 16)
            g = int(HexCode[2:4], 16)
            b = int(HexCode[4:6], 16)
            return [r,g,b]
        return [0,0,0]

    def int_to_RGB(self, int_img):
        # print(f'int_img:  {int_img.shape}')
        rgb_img = np.zeros([256,256,3])
        for part in self.ColorMap:
            id = int(self.ColorMap[part]['id'])
            rgb_img[int_img[0]==id] = self.colorHexToRGB(self.ColorMap[part]['color'])
        rgb_img = np.uint8(rgb_img)
        return rgb_img

    def RGB_to_int(self, rgb_img):
        int_img = np.zeros([256,256,1])
        for part in self.ColorMap:
            color = self.colorHexToRGB(self.ColorMap[part]['color'])
            r = color[0]
            g = color[1]
            b = color[2]
            r_mask = (rgb_img[:,:,0]==r)
            g_mask = (rgb_img[:,:,1]==g)
            b_mask = (rgb_img[:,:,2]==b)
            rgb_mask = r_mask & g_mask & b_mask
            int_img[rgb_mask] = int(self.ColorMap[part]['id'])
        return int_img

    def gen_mask(self, src_name:str, dst_name:str):
        src_path = os.path.join(self.root_path, src_name)
        dst_path = os.path.join(self.root_path, dst_name)
        # read image
        image = Image.open(src_path)
        image = image.resize((256,256))
        image = np.array(image)[:,:,:3]
        image = (image / 127.5) - 1.0
        rgb_image = torch.from_numpy(image).permute(2,0,1).float()
        init_mask = torch.zeros([1,256,256])
        print(f'{rgb_image.shape}')
        clean_images = torch.cat([rgb_image, init_mask], dim=0).unsqueeze(0).cuda()
        A_maskgen = torch.ones(clean_images.shape).cuda()
        A_maskgen[:, 3, :, :] *= 0 # 生成 mask 的时候观测是前三维
        mask_gen = self.inverse(clean_images, A_maskgen)[:, 3, :, :]
        mask_gen_return = mask_gen
        
        mask_gen = np.round(np.clip((mask_gen + 1) * 10, 0, 18))
        
        mask_gen_color = self.int_to_RGB(mask_gen)
        
        mask_gen_color = self.numpy_to_image(mask_gen_color)
        # save image
        mask_gen_color.save(dst_path, 'PNG')
        self.mask_gen = mask_gen
        self.num_mask += 1
        return f'{self.num_mask}'

    def gen_final(self, src_name:str, dst_name:str):
        src_path = os.path.join(self.root_path, src_name)
        dst_path = os.path.join(self.root_path, dst_name)

        rgb_image = Image.open(src_path)
        rgb_image = np.array(rgb_image)
        int_img = self.RGB_to_int(rgb_image) 
        int_img = (int_img / 10.0) - 1.0
        
        mask = torch.from_numpy(int_img).permute(2,0,1).float()
        
        init_rgb = torch.zeros([3, 256, 256])
        clean_images = torch.cat([init_rgb, mask], dim=0).unsqueeze(0).cuda()
        A_maskgen = torch.ones(clean_images.shape).cuda()
        A_maskgen[:, 0:3, :, :] *= 0 # 生成mask的时候观测是前三维
        img_gen = self.inverse(clean_images, A_maskgen)[:, 0:3, :, :]
        
        img_gen = np.round((img_gen+1.0)*127.5)

        # (1,3,256,256)
        img_gen = img_gen.squeeze(0)
        img_gen = torch.from_numpy(img_gen).permute(1,2,0).numpy()
        # print(img_gen.shape)
        # save image
        img_gen = self.numpy_to_image(img_gen)
        img_gen.save(dst_path, 'PNG')

        self.num_final += 1
        return f'{self.num_final}'
if __name__ == '__main__':
    model = DiffusionModel(root_path='/home/zhuangjiaxin/workspace/diffusion/diffusion-demo-website/flask_backend/upload_imgs')
    # model.gen_mask('1.jpg', '1_mask.png')
    model.gen_final('1_mask.png', '1_final.png')