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
        np_array = np_array.astype(np.int8)
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
        rgb_img = np.zeros((256,256,3))
        for part in self.ColorMap:
            id = int(self.ColorMap[part]['id'])
            rgb_img[int_img[0]==id] = self.colorHexToRGB(self.ColorMap[part]['color'])
        rgb_img = np.uint8(rgb_img)
        return rgb_img

    def RGB_to_int(self, rgb_img):
        int_img = np.zeros((256,256))
        for part in self.ColorMap:
            color = self.colorHexToRGB(self.ColorMap[part]['color'])
            id = int(self.ColorMap[part]['id'])
            r_mask = (rgb_img[:,:,0]==color[0])
            g_mask = (rgb_img[:,:,1]==color[1])
            b_mask = (rgb_img[:,:,2]==color[2])
            rgb_mask = r_mask & g_mask & b_mask 
            int_img[rgb_mask] = id
        # int_img = int_img.astype(np.int8)
        # image = Image.fromarray(int_img, 'L')
        # image.save(os.path.join(self.root_path, 'int_img.png'), 'PNG')
        # exit()
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
        self.num_mask += 1
        return f'{self.num_mask}'

    def gen_final(self, src_name:str, dst_name:str, gen_mask:str, canvas_mask:str, mode='modify'):
        # get img pathes
        src_path = os.path.join(self.root_path, src_name)
        dst_path = os.path.join(self.root_path, dst_name)
        gen_mask_path = os.path.join(self.root_path, gen_mask)
        canvas_mask_path = os.path.join(self.root_path, canvas_mask)
        # open images
        raw_image = Image.open(src_path).convert('RGB')
        raw_image = raw_image.resize((256,256))
        gen_mask_image = Image.open(gen_mask_path).convert('RGBA')
        canvas_mask_image = Image.open(canvas_mask_path).convert('RGBA')
        # switch mode
        if mode == 'modify':
            # raw image
            raw_image = np.array(raw_image)[:,:,:3]
            raw_image = torch.from_numpy(raw_image).permute(2,0,1).float()
            # edited mask
            edited_mask_image = Image.alpha_composite(gen_mask_image, canvas_mask_image)
            edited_mask_image = np.array(edited_mask_image)[:,:,:3]
            # mask_img = Image.fromarray(edited_mask_image, 'RGB')
            # mask_img.save(os.path.join(self.root_path, 'edited_mask_image.png'),'PNG')
            # exit()
            edited_mask_image = self.RGB_to_int(edited_mask_image)
            edited_mask_image = np.expand_dims(edited_mask_image, axis=-1)
            print(edited_mask_image.shape)
            edited_mask_image = (edited_mask_image / 10.0) - 1.0    # range [-1,1]
            edited_mask_image = torch.from_numpy(edited_mask_image).permute(2,0,1).float()
            # A
            edited_canvas = np.array(canvas_mask_image)[:,:,3]
            edited_pos = np.nonzero(edited_canvas)
            A = np.zeros_like(edited_canvas,dtype=np.int8)
            A[edited_pos] = 1
            # A = Image.fromarray(A, 'L')
            # A.save(os.path.join(self.root_path, 'A.png'), 'PNG')
            # exit()
            A = np.expand_dims(A, axis=2)
            A = np.repeat(A, 4, axis=2)
            A[:,:,3] = 1
            A = torch.from_numpy(A).permute(2,0,1).float()
            A = A.unsqueeze(0)
            # concate
            print(f'{raw_image.shape},{edited_mask_image.shape}')
            clean_images = torch.cat([raw_image, edited_mask_image], dim=0).unsqueeze(0)
            # inverse
            print(f'{clean_images.shape}, {A.shape}')
            A = A.cuda()
            clean_images = clean_images.cuda()
            new_image = self.inverse(clean_images, A)[0, 0:3, :, :]
            new_image = torch.from_numpy(new_image).permute(1,2,0).numpy()
            new_image = np.round((new_image+1.0)*127.5)
            new_image = self.numpy_to_image(new_image)
            new_image.save(dst_path, 'PNG')
        else:
            pass

        self.num_final += 1
        return f'{self.num_final}'
if __name__ == '__main__':
    model = DiffusionModel(root_path='/home/zhuangjiaxin/workspace/diffusion/diffusion-demo-website/flask_backend/upload_imgs')
    # model.gen_mask('1.jpg', 'mask-img.png')
    model.gen_final('up-img.png', '1_final.png', 'mask-img.png', 'edited_mask.png')
    # def colorHexToRGB(colorHex:str):
    #     if colorHex[0] == '#' and len(colorHex)==7:
    #         HexCode = colorHex[1:]
    #         r = int(HexCode[0:2], 16)
    #         g = int(HexCode[2:4], 16)
    #         b = int(HexCode[4:6], 16)
    #         return [r,g,b]
    #     return [0,0,0]
    # test_img = np.zeros((8,8,3))
    # ColorMap = {
    #     'background':   {'id': 0, 'color': '#000000'},
    #     'skin':         {'id': 1, 'color': '#cc0000'},
    #     'nose':         {'id': 2, 'color': '#4c9900'},
    #     'eye_glasses':  {'id': 3, 'color': '#cccc00'},
    #     'eye_left':     {'id': 4, 'color': '#3333ff'},
    # }
    # root_path='/home/zhuangjiaxin/workspace/diffusion/diffusion-demo-website/flask_backend/upload_imgs'
    # step = 4
    
    # skin_rgb = colorHexToRGB(ColorMap['skin']['color'])
    # nose_rgb = colorHexToRGB(ColorMap['nose']['color'])
    # glass_rgb = colorHexToRGB(ColorMap['eye_glasses']['color'])
    # eye_l_rgb = colorHexToRGB(ColorMap['eye_left']['color'])

    # test_img[0:int(step),0:int(step)] = skin_rgb
    # test_img[0:int(step),int(step):int(2*step)] = nose_rgb
    # test_img[int(step):int(2*step),0:int(step)] = glass_rgb
    # test_img[int(step):int(2*step),int(step):int(2*step)] = eye_l_rgb
    # print(f"{skin_rgb},{nose_rgb},{glass_rgb},{eye_l_rgb}")
    # for i in range(3):
    #     print(test_img[...,i])
    # test_img = test_img.astype(np.int8)
    # test_img = Image.fromarray(test_img,'RGB')
    # test_img.save(os.path.join(root_path,'test.png'), 'PNG')

    # test_img = Image.open(os.path.join(root_path,'test.png'))
    # test_img = np.array(test_img)
    # demo_img = np.zeros((8,8))
    # # rgb_mask = np.all((test_img==skin_rgb))
    # # print(rgb_mask)
    # color = skin_rgb
    # r_mask = (test_img[:,:,0]==color[0])
    # g_mask = (test_img[:,:,1]==color[1])
    # b_mask = (test_img[:,:,2]==color[2])
    # rgb_mask = r_mask & g_mask & b_mask
    # demo_img[rgb_mask] = 255
    # demo_img = demo_img.astype(np.int8)
    # demo_img = Image.fromarray(demo_img,'L')
    # demo_img.save(os.path.join(root_path,'test-1.png'), 'PNG')

