import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt

import torch
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAN_dataset(torch.utils.data.Dataset):
    def __init__(self, path_data, path_target, transform=None):
        self.transform = transform
        data = np.load(path_data)
        target = np.load(path_target)

        N = data.shape[0]

        self.data = data
        self.target = target
        self.len = N

    def __getitem__(self, index):
        if self.transform:
            # because target and input need to share same random transform
            tmp = np.concatenate([self.data[index], self.target[index]], axis=-1)
            tmp = self.transform(tmp)
            return tmp[:4, :, :], tmp[4:, :, :]
        return self.data[index], self.target(self.labels[index])

    def __len__(self):
        return self.len


"""
Func for Validation 
"""


def evaluate(G, G_criterion, loader):
    G.eval()
    errG_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            fake = G(data)

            errG_2 = G_criterion(fake, target)
            errG = errG_2
            errG_list.append(errG.item())

    img = fake[0].detach().to("cpu").numpy()
    data_img = data[0].detach().to("cpu").numpy()
    target_img = target[0].detach().to("cpu").numpy()

    plt.subplot(2, 2, 1)
    plt.title("VALIDATION")
    plt.axis("off")
    plt.imshow(np.transpose(data_img, (1, 2, 0))[:, :, :3][:, :, ::-1])

    plt.subplot(2, 2, 2)
    plt.axis("off")
    plt.imshow(np.transpose(data_img, (1, 2, 0))[:, :, 3])

    plt.subplot(2, 2, 3)
    plt.imshow(np.transpose(img, (1, 2, 0))[:, :, ::-1])
    plt.subplot(2, 2, 4)
    plt.imshow(np.transpose(target_img, (1, 2, 0))[:, :, ::-1])

    plt.show()

    return np.mean(errG_list)


def train(
    light_generator,
    optimizerG,
    schedulerG,
    G_criterion,
    train_loader,
    val_loader,
    num_epochs=41,
    Freq_print=5,
    Freq_plot=5,
):
    G_losses = []
    info = []

    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        light_generator.train()
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            fake = light_generator.forward(data)

            light_generator.zero_grad()

            # Calculate gradients for G
            errG_1 = G_criterion(fake, target)
            errG = errG_1

            # Update G
            errG.backward()
            optimizerG.step()

        if epoch % Freq_print == 0:
            print("epoch = {}, G_loss_MSE = {:.5f}".format(epoch, errG_1.item()))

        # Update lr scheduler
        schedulerG.step()

        if epoch % Freq_plot == 0:
            img = fake[0].detach().to("cpu").numpy()
            data_img = data[0].detach().to("cpu").numpy()
            target_img = target[0].detach().to("cpu").numpy()

            plt.subplot(2, 2, 1)
            plt.title("TRAIN")
            plt.axis("off")
            plt.imshow(np.transpose(data_img, (1, 2, 0))[:, :, :3][:, :, ::-1])

            plt.subplot(2, 2, 2)
            plt.axis("off")
            plt.imshow(np.transpose(data_img, (1, 2, 0))[:, :, 3])

            plt.subplot(2, 2, 3)
            plt.imshow(np.transpose(img, (1, 2, 0))[:, :, ::-1])

            plt.subplot(2, 2, 4)
            plt.imshow(np.transpose(target_img, (1, 2, 0))[:, :, ::-1])

            plt.show()

            val_errG = evaluate(light_generator, G_criterion, val_loader)
            print("#Validation, G_loss = {:.5f}#".format(val_errG))

            info.append([errG.item(), val_errG])
    torch.save(
        {
            "model_state_dict": light_generator.state_dict(),
        },
        "model\light_generator.pt",
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulate_light_circle(G, img, mask_pos):
    # -----mask part-----#
    center = mask_pos[:2]
    radius = mask_pos[-1]
    mask = draw_circle_mask(img.shape[:2], center, radius)
    # -----simulate part-----#
    data = np.insert(img, 3, mask, axis=-1)
    torch_data = torchvision.transforms.ToTensor()(data).unsqueeze(0).to(device)

    G_out = G.eval_forward(torch_data)
    G_out = G_out.to("cpu")

    out_img = G_out[0].detach().numpy()

    return np.transpose(out_img, (1, 2, 0)), mask


def simulate_light_polygon(G, img, mask_pos):
    # ----init part----#
    points = []
    for i in range(len(mask_pos) // 2):
        points.append([mask_pos[i * 2], mask_pos[i * 2 + 1]])
    # points=np.array(points)
    points = sort_clockwise(
        points
    )  # need to follow clockwise or reverse to build edges

    edges = []
    for i in range(0, len(points) - 1, 1):
        edges.append([points[i], points[i + 1]])
    edges.append([points[-1], points[0]])
    # ----polygon mask-----#
    # https://www.youtube.com/watch?v=RSXM9bgqxJM
    H, W = img.shape[:2]
    mask = np.zeros([H, W])
    for i in range(H):
        for j in range(W):
            if is_inside((i, j), edges):
                mask[i, j] = 1
    # ----simulate part----#
    data = np.insert(img, 3, mask, axis=-1)
    torch_data = torchvision.transforms.ToTensor()(data).unsqueeze(0).to(device)

    G_out = G.eval_forward(torch_data)
    G_out = G_out.to("cpu")

    out_img = G_out[0].detach().numpy()
    return np.transpose(out_img, (1, 2, 0)), mask


def simulate_light_elipse(G, img, mask_pos):
    cx, cy, a, b, theta = mask_pos
    # ----elipse mask-----#
    H, W = img.shape[:2]
    mask = np.zeros([H, W])
    for x in range(H):
        for y in range(H):
            dx = x - cx
            dy = y - cy
            # -----rotate---
            rdx = np.cos(theta) * dx - np.sin(theta) * dy
            rdy = np.sin(theta) * dx + np.cos(theta) * dy

            calc = (rdx**2) / (a**2) + (rdy**2) / (b**2)
            if calc < 1:
                mask[x, y] = 1
    # ----simulate part----#
    data = np.insert(img, 3, mask, axis=-1)
    torch_data = torchvision.transforms.ToTensor()(data).unsqueeze(0).to(device)

    G_out = G.eval_forward(torch_data)
    G_out = G_out.to("cpu")

    out_img = G_out[0].detach().numpy()
    return np.transpose(out_img, (1, 2, 0)), mask


def draw_circle_mask(shape, center, radius):
    mask = np.zeros(shape)
    r2 = radius**2
    for i in range(shape[0]):
        for j in range(shape[1]):
            dist = (i - center[0]) ** 2 + (j - center[1]) ** 2
            if dist < r2:
                mask[i, j] = 1
    return mask


def sort_clockwise(points):
    heap = []
    center = np.mean(points, axis=0)
    for p in points:
        vector = p - center
        dx, dy = vector[0], vector[1]

        degree = np.rad2deg(np.arctan(dy / (dx + 0.001)))
        if dx < 0 and dy > 0:
            degree += 180
        elif dx < 0 and dy < 0:
            degree += 180
        elif dx > 0 and dy < 0:
            degree += 360
        heappush(heap, (degree, p))
    sorted_points = [heappop(heap)[1] for i in range(len(heap))]
    return sorted_points


# https://www.youtube.com/watch?v=RSXM9bgqxJM
def is_inside(test_point, edges):
    cnt = 0
    xp, yp = test_point
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp - y1) / (y2 - y1)) * (x2 - x1):
            cnt += 1
    return cnt % 2 == 1
