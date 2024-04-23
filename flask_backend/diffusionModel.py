import os
import shutil

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

    