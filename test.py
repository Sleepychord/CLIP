# %%
import torch
import clip
from PIL import Image
image_resolution = 224
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
# model, preprocess = clip.load("RN50x4", device=device, input_resolution=image_resolution)

# %%
image = preprocess(Image.open("corgi.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "mountain", "corgi", "a dog is playing guitar"]).to(device)

# %%
with torch.no_grad():
    image_features, value_map = model.encode_image(image, output_features=True)
    text_features, text_map, eos_pos = model.encode_text(text, output_features=True)

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    value_map_raw = value_map / value_map.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    text_map = text_map / text_map.norm(dim=1, keepdim=True)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# print("value_map:", value_map[0][0].var(), value_map[0][1].var(), value_map[0][-1].var())
 # %%
import math
import seaborn as sns
import matplotlib.pylab as plt

from PIL import ImageDraw

def visualize(value_map, image_path, image_size=image_resolution, lim=0.2):
    s = image_size // value_map.shape[0]
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(
            value_map.cpu().numpy(), 
            ax=axes,
            linewidth = 0.5,
            cmap = 'coolwarm'
        )
    plt.show()
    if image_path is None:
        return
    img = Image.open(image_path)
    width, height = img.size
    scale = image_size / min(width, height)
    img = img.resize(
                (int(round(scale * width)), int(round(scale * height))),
                resample=Image.Resampling.BICUBIC,
            )
    width, height = img.size   # Get dimensions

    left = (width - image_size)/2
    top = (height - image_size)/2
    right = (width + image_size)/2
    bottom = (height + image_size)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    
    img1 = ImageDraw.Draw(img)
        
    for x in range(value_map.shape[0]):
        for y in range(value_map.shape[0]):
            if value_map[y, x] > lim:
                img1.rectangle((x*s, y*s, x*s+s, y*s+s))
    img.save(f"tests/vis.png")
    
# %%
value_map = (value_map_raw[0] * text_features[4]).sum(dim=-1)
sz = int(math.sqrt(value_map.shape[0]))
value_map = value_map.reshape(sz, sz)
visualize(value_map, 'corgi.png')

# %%
txt_map_id = 4
tm = text_map[txt_map_id]
for token_id in range(eos_pos[txt_map_id]+1):
    print(text[txt_map_id][token_id])
    value_map = (value_map_raw[0] * tm[token_id]).sum(dim=-1)
    sz = int(math.sqrt(value_map.shape[0]))
    value_map = value_map.reshape(sz, sz)
    visualize(value_map, None)
# %%
