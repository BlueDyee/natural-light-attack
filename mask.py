import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

import os
import re
import json
import pandas as pd


def mask_extract(
    source_img,
    target_num,
    if_rotate,
    x_bound,
    y_bound,
    abs_threshold=13,
    k_size=7,
    median_times=1,
    redraw_times=1,
    val_sample_freq=10,
    PATH_data="dataset/physical/",
):
    for ith in range(target_num):
        ###
        PATH_video = "workspace_physical/target_{}.mov".format(ith)
        SHAPE_RESIZED = (128, 128)
        hist_bins, hist_resolution, hist_threshold = 128, 3, 200
        blur_redraw_threshold = 0.35
        cap = cv2.VideoCapture(PATH_video)
        ###

        data = []
        d_L_masks = []
        orig_masks = []
        target = []

        val_data = []
        val_target = []
        iters = 0

        test_img = source_img.astype(np.float32) / 255.0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                iters += 1
                # ---------1.3.1 to Lab------------#
                Lab_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)
                if if_rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = frame[x_bound[0] : x_bound[1], y_bound[0] : y_bound[1]]
                frame = cv2.resize(frame, SHAPE_RESIZED, interpolation=cv2.INTER_CUBIC)
                orig_frame = frame

                frame = frame.astype(np.float32) / 255.0
                frame = cv2.resize(frame, SHAPE_RESIZED, interpolation=cv2.INTER_CUBIC)
                frame = np.clip(frame, 0, 1)
                Lab_img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

                # --------1.3.2 Lab distance----------#
                d_L = np.linalg.norm(Lab_img2.astype(float) - Lab_img, axis=-1)

                # --------1.3.3 Threshold from hist----------#
                if abs_threshold:
                    threshold = abs_threshold
                else:
                    his = np.histogram(d_L, bins=range(0, hist_bins, hist_resolution))
                    threshold = his[1][np.where(his[0] <= hist_threshold)[0][0]]
                # his[1] is the indexes of corresponding hist_interval, [0][0] take our first inerval from 2D array

                # --------1.3.4 Mask scratch from threshold------------#
                light_index = np.where(d_L > threshold)  # output is tuple 2*n
                mask = np.zeros_like(d_L)
                mask[light_index] = 1

                mask = mask.astype("uint8")  # cv2 median filter required uint8
                orig_mask = mask.copy()

                # --------1.3.5 Median out noise----------#
                for i in range(median_times):
                    mask = cv2.medianBlur(mask, k_size)
                mask = mask.astype("float64")  # for redraw

                # --------1.3.6 Use Blur to redraw mask------------#
                for i in range(redraw_times):
                    mask = cv2.blur(
                        mask, (k_size, k_size), borderType=cv2.BORDER_CONSTANT
                    )
                    mask = np.where(mask > blur_redraw_threshold, 1, 0).astype(
                        "float64"
                    )
                mask = mask.astype("uint8")

                # --------1.3.7 Concantenate mask to source image------
                # store uint8
                if iters % val_sample_freq == 0:
                    val_data.append(np.insert(source_img, 3, mask, axis=2))
                    val_target.append(orig_frame)
                else:
                    d_L_masks.append(d_L)
                    orig_masks.append(orig_mask)
                    data.append(np.insert(source_img, 3, mask, axis=2))
                    target.append(orig_frame)
            else:
                break

        cap.release()
        data = np.array(data)
        orig_masks = np.array(orig_masks)
        target = np.array(target)
        print(data.shape)
        print(target.shape)

        indexes = mask_demo(
            orig_masks, data, target, k_size, redraw_times, blur_redraw_threshold
        )

        """
        1.4 Store Data 
        """
        PATH_train_data = PATH_data + "train_data_{}.npy".format(ith)
        PATH_train_target = PATH_data + "train_target_{}.npy".format(ith)
        PATH_val_data = PATH_data + "val_data_{}.npy".format(ith)
        PATH_val_target = PATH_data + "val_target_{}.npy".format(ith)

        np.save(PATH_train_data, data)
        np.save(PATH_train_target, target)
        np.save(PATH_val_data, val_data)
        np.save(PATH_val_target, val_target)


def mask_save(PATH_data="dataset/physical/"):
    """
    1.5 Concate all existing sub-data you want to train together
    concate all _1 _2 _3...-> no
    """
    ###
    PATH_train_data = PATH_data + "train_data_0.npy"
    PATH_train_target = PATH_data + "train_target_0.npy"
    PATH_val_data = PATH_data + "val_data_0.npy"
    PATH_val_target = PATH_data + "val_target_0.npy"
    ###

    total_data = concatenate_data(PATH_train_data)
    np.save(orig_name(PATH_train_data), total_data)

    total_data = concatenate_data(PATH_train_target)
    print("Total target(Train) shape=", total_data.shape)
    np.save(orig_name(PATH_train_target), total_data)

    total_data = concatenate_data(PATH_val_data)
    np.save(orig_name(PATH_val_data), total_data)

    total_data = concatenate_data(PATH_val_target)
    np.save(orig_name(PATH_val_target), total_data)
    print("Total target( Val ) shape=", total_data.shape)


def mask_demo(
    orig_masks, data, target, k_size, redraw_times, blur_redraw_threshold, height=5
):
    ###
    # np.random.seed(0)
    width = 4
    ###
    B, X, Y, C = data.shape

    indexes = np.random.randint(B, size=height)
    cur = 1
    plt.figure(figsize=(10, 10))

    for index in indexes:
        mask = orig_masks[index].astype("uint8")

        plt.subplot(height, width, cur)
        plt.title("index={}".format(index))
        plt.imshow(target[index][:, :, ::-1])
        plt.axis("off")

        plt.subplot(height, width, cur + 1)
        plt.title("diff > threshold")
        plt.imshow(mask)
        plt.axis("off")

        plt.subplot(height, width, cur + 2)
        plt.title("medium blur k_size={}".format(k_size))
        blurred = cv2.medianBlur(mask, k_size)
        plt.imshow(blurred)
        plt.axis("off")

        plt.subplot(height, width, cur + 3)
        plt.title("blur+redraw {} times".format(redraw_times))
        blurred = blurred.astype("float64")
        for i in range(redraw_times):
            blurred = cv2.blur(
                blurred, (k_size, k_size), borderType=cv2.BORDER_CONSTANT
            )
            blurred = np.where(blurred > blur_redraw_threshold, 1, 0).astype("float64")
        plt.imshow(blurred)
        plt.axis("off")

        cur += width

    return indexes


def next_name(file_name):
    """
    "dataset/val_data_0.npy" --> "dataset/val_data_1.npy"
    """
    pattern_num = re.compile("[\d]+")
    file_num = re.search(pattern_num, file_name).group()
    replace_num = str(int(file_num) + 1)
    return re.sub(pattern_num, replace_num, file_name, 1)  # 1 replace first pattern


def concatenate_data(file_name):
    data_list = []
    while os.path.exists(file_name):
        data = np.load(file_name)
        data_list.append(data)
        file_name = next_name(file_name)
    return np.concatenate(data_list, axis=0)


def orig_name(file_name):
    """
    "dataset/val_data_0.npy"..... --> "dataset/val_data.npy"
    """
    pattern = re.compile("_\d+")
    return re.sub(pattern, "", file_name, 1)
