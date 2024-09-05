# -*- coding: utf-8 -*-



import os
import numpy as np
import logging


class BaseGenerator(object):
    def __init__(self, config, mode = "train"):    
        self.config = config
        self.mode = mode
        self.num_ad = config.num_ad
        self.num_org = config.num_org
        self.num_bid = config.num_bid
        self.num_vol = config.num_vol
        self.num_agent = config.num_agent
        self.num_item = config.num_item
        self.num_bundle = config.num_bundle
        self.num_instances = config[self.mode].num_batches * config[self.mode].batch_size
        self.num_misreports = config[self.mode].num_misreports
        self.batch_size = config[self.mode].batch_size

        self.log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''

    def build_generator(self, AD=None, ORG=None, VOL=None, ADV=None):
        if self.mode is "train":            
            if self.config.train.data is "fixed":
                if self.config.train.restore_iter == 0:
                    self.get_data(AD, ORG, VOL, ADV)
                else:
                    self.load_data_from_file(self.config.train.restore_iter)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()
                
        else:
            if self.config[self.mode].data is "fixed" or AD is not None or ORG is not None:
                self.get_data(AD, ORG, VOL, ADV)
                self.gen_func = self.gen_fixed()
            else:
                self.get_data(AD, ORG, VOL, ADV)
                self.gen_func = self.gen_fixed()

    def get_data(self, AD=None, ORG=None, VOL=None, ADV=None):
        """ Generates data """

        # 确保 num_i、num_j、num_b 和 num_agents 都设置为相同的值
        num_ad = self.config.num_ad         # 广告数量
        num_org = self.config.num_org       # organic数量
        num_bid = self.config.num_bid       # 报价数量
        num_vol = self.config.num_vol       # volume数量
        num_agent = self.config.num_agent   # 所有节点数量
        num_item = self.config.num_item     # 所有广告位数量
        num_bundle = self.config.num_bundle # 所有bundle的数量
        num_instances = self.num_instances  # 样本总数

        # 定义各种矩阵的形状
        ad_shape = [self.num_instances, num_ad, num_item]
        org_shape = [self.num_instances, num_org, num_item]
        vol_shape = [self.num_instances, num_vol, num_item]
        graph_shape = [self.num_instances, num_bid, num_vol]
        adv_shape = [self.num_misreports, num_instances, num_bid, 1]

        # 生成默认随机数据（如果未提供）
        if AD is None:
            AD = self.generate_ad(ad_shape)

        if ORG is None:
            ORG = self.generate_org(org_shape)

        BID = np.concatenate((AD, ORG), axis=1)

        if ADV is None:
            ADV = self.generate_random_ADV(adv_shape, ad_shape, org_shape, vol_shape)

        if VOL is None:
            VOL = self.generate_volume(vol_shape)
            np.save(os.path.join(self.config.dir_name, 'train_vol' + self.log_suffix + '.npy'), VOL)

        # 生成随机的 tu 和 tu_1 矩阵
        graph_raw, graph_ranked = self.generate_random_tu(graph_shape)

        # 检查并修改 X 矩阵
        for ins in range(num_instances):
            # 检查 tu_ 中每一行是否全为零，如果是，则对应 X 行全为零
            for idx_ad in range(num_ad):
                if np.all(graph_raw[ins, idx_ad, :] == 0):
                    print("_______________________________")
                    BID[ins, idx_ad, :] = 0

            # 检查 tu_ 中每一列是否全为零，如果是，则对应 X 列全为零
            for idx_org in range(num_org):
                if np.all(graph_raw[ins, :, idx_org] == 0):
                    print("_______________________________")
                    BID[ins, num_ad + idx_org, :] = 0

        # 将生成的矩阵保存为类的属性
        self.AD = AD
        self.ORG = ORG
        self.BID = BID
        self.VOL = VOL
        self.ADV = ADV
        self.graph_raw = graph_raw
        self.graph_ranked = graph_ranked

        # 初始化并填充 bundle_item_pair 矩阵
        bundle_item_pair = np.zeros((num_instances, num_bundle * num_item, num_agent * num_item))

        for batch_idx in range(num_instances):
            for j in range(num_agent * num_item):
                hh = j % num_item
                ww = int(j / num_item)


                if ww < num_bid:  # gsp
                    for k in range(num_vol):
                        # print("ww: ", ww)
                        # print("hh: ", hh)
                        # print("k: ", k)
                        if self.graph_ranked[batch_idx][ww][k] != 0:
                            cc = self.graph_ranked[batch_idx][ww][k] - 1  # 边序号 - 1
                            uu = int(cc * num_item + hh)  # 前面有几条边 * 广告位 + 竞争的广告位序号 = 真正的边序号

                            bundle_item_pair[batch_idx][uu][j] = 1
                else:
                    for k in range(num_bid):
                        # print("ww: ", ww)
                        # print("hh: ", hh)
                        # print("k: ", k)
                        if self.graph_ranked[batch_idx][k][ww - num_bid] != 0:
                            cc = self.graph_ranked[batch_idx][k][ww - num_bid] - 1
                            uu = int(cc * num_item + hh)
                            bundle_item_pair[batch_idx][uu][j] = 1

        # print("bundle_item_pair: ", bundle_item_pair)
        # 将 xx 矩阵保存为类的属性
        self.bundle_item_pair = bundle_item_pair

    def load_data_from_file(self, iter):
        """ Loads data from disk """
        self.BID = np.load(os.path.join(self.config.dir_name, 'BID.npy'))
        self.ADV = np.load(os.path.join(self.config.dir_name,'ADV_' + str(iter) + '.npy'))
        print("self.config.dir_name",self.config.dir_name)
    def save_data(self, iter):
        """ Saved data to disk """
        if self.config.save_data is False: return
        
        if iter == 0:
            np.save(os.path.join(self.config.dir_name, 'BID'), self.BID)
        else:
            np.save(os.path.join(self.config.dir_name,'ADV_' + str(iter)), self.ADV)            
                       
    def gen_fixed(self):
        i = 0
        if self.mode is "train": perm = np.random.permutation(self.num_instances)
        else: perm = np.arange(self.num_instances)
            
        while True:
            idx = perm[i * self.batch_size: (i + 1) * self.batch_size]

            yield self.AD[idx], self.ORG[idx], self.BID[idx], self.VOL[idx], self.graph_raw[idx], self.graph_ranked[idx], self.bundle_item_pair[idx], self.ADV[:, idx, :, :], idx
            i += 1
            if(i * self.batch_size == self.num_instances):
                i = 0
                if self.mode is "train": perm = np.random.permutation(self.num_instances)
                else: perm = np.arange(self.num_instances)
            
    def gen_online(self):
        ad_batch_shape = [self.batch_size, self.num_ad, self.num_item]
        org_batch_shape = [self.batch_size, self.num_org, self.num_item]
        vol_batch_shape = [self.batch_size, self.num_vol, self.num_item]

        adv_batch_shape = [self.num_misreports, self.batch_size, self.num_agents, 1]

        graph_shape = [self.batch_size, self.num_bid, self.config.num_vol]

        while True:
            AD = self.generate_ad(ad_batch_shape)
            ORG = self.generate_org(org_batch_shape)
            BID = np.concatenate((AD, ORG), axis=1)
            VOL = self.generate_volume(vol_batch_shape)
            ADV = self.generate_random_ADV(adv_batch_shape, ad_batch_shape, org_batch_shape, vol_batch_shape)
            graph_raw, graph_ranked = self.generate_random_tu(tu_shape,self.config.num_bundle)

            bundle_item_pair = np.zeros((self.batch_size, self.config.num_bundle * self.config.num_item,
                                self.config.num_agent * self.config.num_item))

            for batch_idx in range(self.batch_size):
                for j in range(self.config.num_agent * self.config.num_item):
                    hh = j % self.config.num_item
                    ww = int(j / self.config.num_item)
                    if ww < self.config.num_bid:  # gsp
                        for k in range(self.config.num_vol):
                            if graph_ranked[batch_idx][ww][k] != 0:
                                cc = graph_ranked[batch_idx][ww][k] - 1
                                uu = int(cc * self.config.num_item + hh + 1 - 1)
                                bundle_item_pair[batch_idx][uu][j] = 1
                    else:
                        for k in range(self.config.num_bid):
                            if graph_ranked[batch_idx][k][ww - self.config.num_bid] != 0:  ###
                                cc = graph_ranked[batch_idx][k][ww - self.config.num_bid] - 1
                                uu = int(cc * self.config.num_items + hh + 1 - 1)
                                bundle_item_pair[batch_idx][uu][j] = 1

            yield AD, ORG, BID, VOL, graph_raw, graph_ranked, bundle_item_pair, ADV, None

    def update_adv(self, idx, adv_new):
        """ Updates ADV for caching """
        self.ADV[:, idx, :, :] = adv_new

    def generate_ad(self, shape):
        """ Rewrite this for new distributions """
        raise NotImplementedError

    def generate_volume(self, shape):
        """ Rewrite this for new distributions """
        raise NotImplementedError

    def generate_org(self, shape):
        """ Rewrite this for new distributions """
        raise NotImplementedError

    def generate_random_X(self, shape):
        """ Rewrite this for new distributions """
        raise NotImplementedError

    def generate_random_tu(self, shape, bianshu):
        """ Rewrite this for new distributions """
        raise NotImplementedError

    def generate_random_ADV(self, ad_shape, org_shape, vol_shape):
        """ Rewrite this for new distributions """
        raise NotImplementedError