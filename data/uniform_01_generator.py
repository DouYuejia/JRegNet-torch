
# -*- coding: utf-8 -*-
import os

import numpy as np
import random
import tensorflow as tf

from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, AD=None, ORG=None, VOL=None, ADV=None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(AD=AD, ORG=ORG, VOL=VOL, ADV=ADV)

    def generate_uniform(self, shape, low_bound, high_bound):
        tmp_result = np.empty(shape)
        uniform_values = np.random.uniform(low_bound, high_bound, size=(shape[0], shape[1]))
        return uniform_values

    def generate_org(self, shape):
        # 生成0-1均匀分布的值
        uniform_values = self.generate_uniform(shape, low_bound=0, high_bound=0.1)
        # 按照广告拍卖的逻辑，在每个batch上生成bidder的vol的报价，并复制给不同的广告位
        tmp_result = np.tile(np.reshape(uniform_values, [shape[0], shape[1], -1]), [1, 1, shape[-1]])
        # 根据广告商的报价，与广告位的CTR相乘，得到slot-wised bid
        result = tmp_result * np.tile(self.config.CTR.reshape(1, 1, shape[-1]), [shape[0], shape[1], 1])
        return result

    def generate_ad(self,shape):
        # 生成0-1均匀分布的值
        uniform_values = self.generate_uniform(shape, low_bound=0, high_bound=1)
        # 按照广告拍卖的逻辑，在每个batch上生成广告商的报价，并复制给不同的广告位
        tmp_result = np.tile(np.reshape(uniform_values, [shape[0], shape[1], -1]), [1, 1, shape[-1]])
        # 根据广告商的报价，与广告位的CTR相乘，得到slot-wised bid
        result = tmp_result * np.tile(self.config.CTR.reshape(1, 1, shape[-1]), [shape[0], shape[1], 1])
        return result

    def generate_volume(self, shape):
        # np.random.seed(12)
        # 生成0-1均匀分布的值
        uniform_values = np.array([0.6, 0.5, 0.4, 0.8, 0.7])
        # 按照广告拍卖的逻辑，在每个batch上生成广告商的报价，并复制给不同的广告位
        tmp_result = np.tile(np.reshape(uniform_values, [1, self.config.num_vol, -1]), [shape[0], 1, shape[-1]])
        # 根据广告商的报价，与广告位的CTR相乘，得到slot-wised bid
        result = tmp_result * np.tile(self.config.CTR.reshape(1, 1, shape[-1]), [shape[0], shape[1], 1])
        print("volume: ",result[0])
        return result


    def generate_gaussian_array(self, shape):
        pass

    def generate_random_tu(self, shape):
        matrix = np.zeros(shape)
        matrix1 = np.zeros(shape)

        for b in range(shape[0]):
            # 生成对角矩阵
            for i in range(min(shape[1], shape[2])):
                matrix[b, i, i] = 1
                matrix1[b, i, i] = i + 1  # 对角线位置填充顺序数

        return matrix, matrix1

    def generate_random_ADV(self, adv_shape, ad_shape, org_shape, vol_shape):
        ad_ADV = np.tile(np.reshape(self.generate_uniform(ad_shape, 0, 1), [1, ad_shape[0], ad_shape[1], -1]), [adv_shape[0], 1, 1, 1])
        org_ADV = np.tile(np.reshape(self.generate_uniform(org_shape, 0, 0.1),[1, org_shape[0], org_shape[1], -1]), [adv_shape[0], 1, 1, 1])
        vol_ADV = np.tile(np.reshape(self.generate_uniform(vol_shape, 0, 1), [1, vol_shape[0], vol_shape[1], -1]), [adv_shape[0], 1, 1, 1])
        values = np.concatenate([ad_ADV, org_ADV], axis=2)

        return values