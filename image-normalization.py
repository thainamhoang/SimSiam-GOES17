import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
from tqdm import tqdm
import time

MAIN_PATH = "./images"


stack = []
for im in tqdm(sorted(os.listdir(MAIN_PATH))):
    stack.append(torch.from_numpy(plt.imread(f"{MAIN_PATH}/{im}")).to(dtype=torch.float32)/255.)

imgs = torch.stack(stack, dim=3)
print(f"Size: {imgs.size()}")

batch_mean = imgs.view(3, -1).mean(dim=1)
batch_std = imgs.view(3, -1).std(dim=1)
print(f"Mean: {batch_mean}, std: {batch_std}")
