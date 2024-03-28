# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 下午3:20
# @Author  : WangShuYi
# @Email   : 88888888@wangshuyi.cn
# @File    : model_helper.py
import cv2
import numpy as np
import torch
from iopaint.schema import InpaintRequest
from iopaint.model_manager import ModelManager


class ImageProcessor:
    def __init__(self):
        """
        初始化ImageProcessor类

        :param model_name: 模型名称
        :type model_name: str
        :param device: 设备
        :type device: torch.device
        :param no_half: 是否不使用半精度浮点数
        :type no_half: bool
        :param low_mem: 是否低内存模式
        :type low_mem: bool
        :param disable_nsfw_checker: 是否禁用NSFW检查器
        :type disable_nsfw_checker: bool
        :param local_files_only: 是否仅使用本地文件
        :type local_files_only: bool
        :param cpu_textencoder: 是否使用CPU文本编码器
        :type cpu_textencoder: bool
        :param cpu_offload: 是否使用CPU卸载
        :type cpu_offload: bool
        """
        self.model_name: str = 'lama'
        self.device: torch.device = torch.device('cuda:0')
        self.no_half: bool = False
        self.low_mem: bool = False
        self.disable_nsfw_checker: bool = False
        self.local_files_only: bool = False
        self.cpu_textencoder: bool = False
        self.cpu_offload: bool = False
        self.model_manager = ModelManager(
            name=self.model_name,
            device=self.device,
            no_half=self.no_half,
            low_mem=self.low_mem,
            disable_nsfw=self.disable_nsfw_checker,
            sd_cpu_textencoder=self.cpu_textencoder,
            local_files_only=self.local_files_only,
            cpu_offload=self.cpu_offload,
            callback=None,
        )

    def process_image(self, input_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        处理图像

        :param input_img: 输入图像
        :type input_img: np.ndarray
        :param mask: mask图像
        :type mask: np.ndarray
        :return: 处理后的图像
        :rtype: np.ndarray
        """
        inpaint_request = InpaintRequest()

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        output = self.model_manager(input_img, mask, inpaint_request)
        return output


if __name__ == '__main__':
    image_processor = ImageProcessor()
    input_path = 'test_image/raw_input.jpg'
    mask_path = 'test_image/mask.png'
    input_img = cv2.imread(input_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    output_img = image_processor.process_image(input_img, mask)
    cv2.namedWindow('output_img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output_img', 800, 800)
    cv2.imshow('output_img', input_img)
    cv2.waitKey(0)
    cv2.imshow('output_img', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()