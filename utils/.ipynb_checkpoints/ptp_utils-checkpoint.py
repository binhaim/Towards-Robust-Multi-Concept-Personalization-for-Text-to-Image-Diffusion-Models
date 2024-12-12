import abc
import copy
import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List

from diffusers.models.attention import Attention as CrossAttention

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img




class AttendExciteCrossAttnProcessor:

    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        
        # print(f"hidden_states:{hidden_states.shape}")
        
        if encoder_hidden_states is not None: #cross attention인 경우
            # print(f"encoder_hidden_states:{encoder_hidden_states.shape}")
            
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size = batch_size)

            query = attn.to_q(hidden_states)
    
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = attn.to_k(encoder_hidden_states)
            
            value = attn.to_v(encoder_hidden_states)
            
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
    
            attention_probs = self.attnstore(attention_probs, is_cross, self.place_in_unet)
    
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
    
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
                

        else:
        
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size = batch_size)
    
            query = attn.to_q(hidden_states)
    
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            
            # print(f"query:{query.shape}")
            # print(f"key:{key.shape}")
            # print(f"value:{value.shape}")
            
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
    
            # print(f"query:{query.shape}")
            # print(f"key:{key.shape}")
            # print(f"value:{value.shape}")
        
            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            # print(f"attention_probs: {attention_probs.shape}")
    
            # hidden_states:torch.Size([1, 4096, 320])
            # encoder_hidden_states:torch.Size([1, 77, 768])
            # query:torch.Size([1, 4096, 320])
            # key:torch.Size([1, 77, 320])
            # value:torch.Size([1, 77, 320])
            # query:torch.Size([8, 4096, 40])
            # key:torch.Size([8, 77, 40])
            # value:torch.Size([8, 77, 40])
            # attention_probs: torch.Size([8, 4096, 77])

            attention_probs = self.attnstore(attention_probs, is_cross, self.place_in_unet)

    
            hidden_states = torch.bmm(attention_probs, value)
            # print(f"hidden_states after bmm : {hidden_states.shape}")
            hidden_states = attn.batch_to_head_dim(hidden_states)
            # print(f"hidden_states after bthd : {hidden_states.shape}")
    
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # print(f"hidden_states after to_out : {hidden_states.shape}")
            # hidden_states after bmm : torch.Size([8, 4096, 40])
            # hidden_states after bthd : torch.Size([1, 4096, 320])
            # hidden_states after to_out : torch.Size([1, 4096, 320])

        
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(model, controller): #모델의 각 주의 레이어에 controller를 등록해서 상호작용하도록 하는것

    attn_procs = {} 
    cross_att_count = 0
    for name in model.unet.attn_processors.keys(): #모델의 주의 프로세서 키 탐색
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet #processor 에 attentioon store도 같이 넣어서 전달!
        )

    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.replace = False
        self.only_cross = False
        self.d_c, self.d_s, self.m_c, self.m_s, self.u_c, self.u_s = 0, 0, 0, 0, 0, 0


class EmptyControl(AttentionControl):
 
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        # 카운터를 딕셔너리로 관리
        counter_map = {
            "down_cross": "d_c",
            "down_self": "d_s",
            "mid_cross": "m_c",
            "mid_self": "m_s",
            "up_cross": "u_c",
            "up_self": "u_s"
        }

        if self.replace:
            # 해당하는 카운터 증가
            if key in counter_map:
                counter_attr = counter_map[key]
                setattr(self, counter_attr, getattr(self, counter_attr) + 1)

                # cur_step에 따른 조건 확인 및 반환
                if self.only_cross:
                    if "self" in key:
                        return attn
                    elif "cross" in key:
                        # return 0.5*self.attention_store[key][getattr(self, counter_attr) - 1] + 0.5*attn
                        store = self.attention_store[key][getattr(self, counter_attr) - 1]
                        # 기존 값과 새로운 값을 0.5씩 섞어서 할당 

                        return store
                
                else:
                    # return 0.5*self.attention_store[key][getattr(self, counter_attr) - 1] + 0.5*attn
                    return self.attention_store[key][getattr(self, counter_attr) - 1]
            else:
                return attn
        else:
            if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
                self.step_store[key].append(attn)
        return attn

        return attn

    def store_global(self):
        # attention_store의 key 중 'cross'가 포함된 것만 deepcopy한 후 CPU로 이동
        deepcopied_store = copy.deepcopy(self.attention_store)
        
        # 'cross'가 key에 포함된 것만 저장
        cpu_step_store = {key: [item.cpu() for item in deepcopied_store[key]] for key in deepcopied_store if "cross" in key}
        # cpu_step_store = {key: [item for item in deepcopied_store[key]] for key in deepcopied_store if "cross" in key}
        
        self.global_store.append(cpu_step_store)  # CPU로 이동된 복사본을 저장
        print(f"len of global_store is {len(self.global_store)}")

    def between_steps(self):
        if not self.replace:
            self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        if not self.replace:
            self.step_store = self .get_empty_store()
        self.d_c, self.d_s, self.m_c, self.m_s, self.u_c, self.u_s = 0, 0, 0, 0, 0, 0

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention
        
    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = []

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super().__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = []
        self.curr_step_index = 0


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int,
                        is_global = False,
                        step = 0) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    if is_global:
        attention_maps = attention_store.global_store[step]
    else:
        attention_maps = attention_store.get_average_attention()

    # print(attention_maps)
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def get_attention_maps(attention_store: AttentionStore,
                        from_where: List[str],
                        res: int,
                        is_cross: bool) -> torch.Tensor:
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    out = []
    #모든 attention layer에서 맵들을 가져옴
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                out.append(item.sum(0)/item.shape[0])
            #multi head attention의 값들의 평균 
    return out
