import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import pandas as pd


class Point:
    def __init__(self, min_list, max_list, grad_step):
        self.dims = len(max_list)
        self.grad_step = grad_step
        self.max_list = max_list
        self.min_list = min_list
        self.pos = np.array(
            [
                np.random.uniform(min_list[dim], max_list[dim], 1).item()
                for dim in range(self.dims)
            ]
        )

        self.best_pos = self.pos
        self.value = 0
        self.all_value = []
        self.all_pos = []
        self.all_grad = []

    def random_start(self):
        return np.array(
            [
                np.random.uniform(self.min_list[dim], self.max_list[dim], 1).item()
                for dim in range(self.dims)
            ]
        )


class ZeroOrderOptimizer:
    def __init__(
        self,
        img,
        iter,
        label,
        attack_model,
        light_generator,
        criteria,
        point_num,
        draw_light,
        min_list,
        max_list,
        classifier_input_shape,
        LABELS,
        grad_step=1,
        lr_step=3,
        threshold_restart=0.00001,
        physical_attack=False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img = img
        self.label = label.to(self.device)
        self.iter = iter
        self.attack_model = attack_model
        self.G = light_generator

        self.criteria = criteria
        self.draw_light = draw_light

        self.max_list = max_list
        self.min_list = min_list
        self.dims = len(max_list)
        self.member_list = [
            Point(min_list, max_list, grad_step=grad_step) for _ in range(point_num)
        ]

        self.classifier_input_shape = classifier_input_shape
        self.LABELS = LABELS
        self.lr_step = lr_step
        self.threshold_restart = threshold_restart
        self.physical_attack = physical_attack
        self.global_best_value = -1
        self.attack_img = img
        # statistic
        self.mislead_table = np.zeros(len(self.LABELS))
        self.mislead_pos = []

        self.initialize()

    def light_inference(self, pos):
        att_img, _ = self.draw_light(self.G, self.img, pos)

        resize_att_img = cv2.resize(
            att_img, self.classifier_input_shape, interpolation=cv2.INTER_AREA
        )
        tensor_img = torchvision.transforms.ToTensor()(resize_att_img).to(self.device)
        tensor_img = tensor_img.unsqueeze(0)
        logits = self.attack_model.forward(tensor_img)
        index = torch.argmax(logits[0]).item()
        value = self.criteria(logits, self.label.unsqueeze(0)).to("cpu").item()

        if value > self.global_best_value:
            self.attack_img = att_img
            self.global_best_value = value
            self.global_best_pos = pos
            self.predict = index

            probs = torch.softmax(logits.squeeze(), dim=-1)
            self.confidence = probs[index]
            if self.label != index:
                self.mislead_table[index] += 1
                self.mislead_pos.append(pos)

        return att_img, index, value, logits[0][index]

    def initialize(self):
        best_index = -1
        best_value = 0
        demo_list = []
        demo_index_list = []
        for i in range(len(self.member_list)):
            with torch.no_grad():
                cur_member = self.member_list[i]
                att_img, index, value, confidence = self.light_inference(cur_member.pos)

                cur_member.value = value
                # print(cur_member.value)
                demo_list.append(att_img)
                demo_index_list.append(index)

        # self.demo(demo_list, demo_index_list)

    def demo(self, demo_list, demo_index_list):
        height = int(len(demo_list) ** 0.5)
        width = len(demo_list) // height
        cur = 0
        # plt.figure(figsize=(12,12))
        plt.subplots_adjust(left=0, bottom=0, top=1, right=1, hspace=0, wspace=0.1)

        for i in range(height):
            for j in range(width):
                plt.subplot(height, width, cur + 1)
                plt.title(
                    "{}...".format(self.LABELS[demo_index_list[cur]][:25]), fontsize=8
                )
                plt.imshow(demo_list[cur][:, :, ::-1])
                plt.axis("off")
                cur += 1

        plt.show()
        return

    def find_grad(self, member, axis=0):
        d_pos = member.pos.copy()
        d_pos[axis] += member.grad_step
        att_img, index, value, confidence = self.light_inference(d_pos)

        return value - member.value

    def update_pos(self, member):
        with torch.no_grad():
            grad = []
            for dim in range(self.dims):
                dx = self.find_grad(member, axis=dim)
                grad.append(dx)
            grad_norm = np.linalg.norm(np.array(grad))  # for normalize to unit vector

            member.all_grad.append(grad_norm)
            if grad_norm > self.threshold_restart:
                step = self.lr_step * (np.array(grad) / grad_norm)
                member.pos = member.pos + step
                # Clip follow given max_list
                member.pos = [
                    member.pos[dim]
                    .clip(self.min_list[dim], self.max_list[dim])
                    .astype(np.int16)
                    for dim in range(self.dims)
                ]

            else:
                member.pos = member.random_start()

            att_img, index, value, confidence = self.light_inference(member.pos)
            member.value = value
        """
        if member.value > self.global_best_value:
            self.attack_img = att_img
            self.global_best_value = member.value
            self.global_best_pos = member.pos
            self.predict = index
            self.confidence = confidence

        if index != self.label:
            self.success = True
            ###
            # statistic
            self.mislead_pos.append((member.pos, index))
            self.mislead_table[index] += 1
            ###
        """

    def run(self):
        for _ in range(self.iter):
            for i in range(len(self.member_list)):
                self.update_pos(self.member_list[i])


class Circle:
    def __init__(self, max_list, min_list, max_speed=3):
        self.dims = len(max_list)
        self.max_list = max_list
        self.min_list = min_list
        self.pos = np.array(
            [
                np.random.uniform(min_list[dim], max_list[dim], 1).item()
                for dim in range(self.dims)
            ]
        )
        self.best_pos = self.pos
        self.speed = np.random.uniform(-max_speed, max_speed, self.dims)

        self.value = 0
        self.best_value = 0


class PSO:
    def __init__(
        self,
        type,
        classifier,
        generator,
        criteria,
        draw_light,
        img,
        label,
        max_list,
        min_list,
        iter=20,
        members=5,
        max_speed=3,
        cognition_factor=0.5,
        social_factor=0.5,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if type == "circle":
            self.member_list = [
                Circle(max_list, min_list, max_speed=max_speed) for _ in range(members)
            ]
        self.type = "circle"
        self.dims = len(max_list)

        self.img = img
        self.attack_img = img
        self.label = label.to(self.device)
        self.attack_model = classifier.to(self.device)
        self.G = generator
        self.criteria = criteria
        self.draw_light = draw_light
        self.iter = iter
        self.members = members
        self.max_list = max_list
        self.min_list = min_list
        self.max_speed = max_speed
        self.cognition_factor = cognition_factor
        self.social_factor = social_factor

        self.EOT = False
        self.success = False
        self.queries = 0
        self.global_best_value, self.global_best_pos = self.initialize()

    def light_inference(self, pos):
        att_img, _ = self.draw_light(self.G, self.img, pos[0:2], pos[2])

        resize_att_img = cv2.resize(att_img, (32, 32), interpolation=cv2.INTER_AREA)
        tensor_img = torchvision.transforms.ToTensor()(resize_att_img).to(self.device)
        tensor_img = tensor_img.unsqueeze(0)
        predict = self.attack_model.forward(tensor_img)
        predict = torch.exp(predict[0])

        index = torch.argmax(predict).item()

        value = self.criteria(predict, self.label).to("cpu").item()
        return att_img, index, value, predict[index]

    def initialize(self):
        best_index = 0
        best_value = 0
        self.attack_model.eval()
        for i in range(len(self.member_list)):
            with torch.no_grad():
                att_img, index, value, confidence = self.light_inference(
                    self.member_list[i].pos
                )
                self.member_list[i].best_value = value
                if self.member_list[i].best_value > best_value:
                    self.attack_img = att_img
                    best_value = self.member_list[i].best_value
                    best_index = i
                    self.predict = index
                    self.confidence = confidence
        return best_value, self.member_list[best_index].pos

    def update_speed(self, member):
        speed = (
            (0.5 + np.random.uniform(self.dims)) * member.speed
            + self.cognition_factor
            * np.random.uniform(self.dims)
            * (np.array(member.best_pos) - np.array(member.pos))
            + self.social_factor
            * np.random.uniform(self.dims)
            * (np.array(self.global_best_pos) - np.array(member.pos))
        )
        member.speed = speed.clip(-self.max_speed, self.max_speed)

    def update_pos(self, member):
        with torch.no_grad():
            member.pos = member.pos + member.speed
            member.pos = [
                member.pos[dim]
                .clip(self.min_list[dim], self.max_list[dim])
                .astype(np.int16)
                for dim in range(self.dims)
            ]
            att_img, index, value, confidence = self.light_inference(member.pos)
            member.value = value

        if value > member.best_value:
            member.best_value = value
            member.best_pos = member.pos

        if value > self.global_best_value:
            self.attack_img = att_img
            self.global_best_value = value
            self.global_best_pos = member.pos
            self.predict = index
            self.confidence = confidence

        if index != self.label:
            self.success = True

    def run(self):
        for _ in range(self.iter):
            for i in range(len(self.member_list)):
                self.update_speed(self.member_list[i])
                self.update_pos(self.member_list[i])
