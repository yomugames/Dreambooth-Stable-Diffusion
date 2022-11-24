
# coding: utf-8

import argparse
import json

# # Dreambooth
# ### Notebook implementation by Joe Penna (@MysteryGuitarM on Twitter) - Improvements by David Bielejeski
# 
# ### Instructions
# - Sign up for RunPod here: https://runpod.io/?ref=n8yfwyum
#     - Note: That's my personal referral link. Please don't use it if we are mortal enemies.
# 
# - Click *Deploy* on either `SECURE CLOUD` or `COMMUNITY CLOUD`
# 
# - Follow the rest of the instructions in this video: https://www.youtube.com/watch?v=7m__xadX0z0#t=5m33.1s
# 
# Latest information on:
# https://github.com/JoePenna/Dreambooth-Stable-Diffusion

# # Upload your training images
# Upload 10-20 images of someone to
# 
# ```
# /content/Dreambooth-Stable-Diffusion/training_images
# ```
# 
# WARNING: Be sure to upload an *even* amount of images, otherwise the training inexplicably stops at 1500 steps.
# 
# *   2-3 full body
# *   3-5 upper body
# *   5-12 close-up on face
# 
# The images should be:
# 
# - as close as possible to the kind of images you're trying to make

# In[ ]:

parser = argparse.ArgumentParser(description="Dreambooth training script.")
parser.add_argument(
    "--session",
    type=str,
    help="the session name for dreambooth train + inference",
    required=True
)

parser.add_argument(
    "--s3images",
    type=str,
    help="the array of images for dreambooth to train"
)

opt = parser.parse_args()

Session_Name = opt.session #@param{type: 'string'}
urls = json.loads(opt.s3images)

seed='332' #@param{type: 'string'}

Session_Name=Session_Name.replace(" ","_")

SessionFolder = '/content/gdrive/MyDrive/Fast-Dreambooth/Sessions/' + Session_Name
InstanceImagesFolder = SessionFolder + '/instance_images'
OutputFolder = SessionFolder + '/output'

#@markdown Add here the URLs to the images of the subject you are adding



#@title Download and check the images you have just added
import os
import requests
from io import BytesIO
from PIL import Image


def image_grid(imgs, rows, cols):
 assert len(imgs) == rows*cols

 w, h = imgs[0].size
 grid = Image.new('RGB', size=(cols*w, rows*h))
 grid_w, grid_h = grid.size

 for i, img in enumerate(imgs):
  grid.paste(img, box=(i%cols*w, i//cols*h))
 return grid

def download_image(url):
 try:
  response = requests.get(url)
 except:
  return None
 return Image.open(BytesIO(response.content)).convert("RGB")

images = list(filter(None,[download_image(url) for url in urls]))
save_path = InstanceImagesFolder
if not os.path.exists(save_path):
 os.mkdir(save_path)
[image.save(f"{save_path}/{i}.png", format="png") for i, image in enumerate(images)]
image_grid(images, 1, len(images))


# ## Training
# 
# If training a person or subject, keep an eye on your project's `logs/{folder}/images/train/samples_scaled_gs-00xxxx` generations.
# 
# If training a style, keep an eye on your project's `logs/{folder}/images/train/samples_gs-00xxxx` generations.

# In[ ]:


# Training

# This isn't used for training, just to help you remember what your trained into the model.
project_name = "project_name"

# MAX STEPS
# How many steps do you want to train for?
max_training_steps = len(images) * 100   #2700 #@param{type: 'number'}

# Match class_word to the category of the regularization images you chose above.
class_word = "person" # typical uses are "man", "person", "woman"

# This is the unique token you are incorporating into the stable diffusion model.
token = "userxyz"


reg_data_root = "/content/Dreambooth-Stable-Diffusion/regularization_images/" + dataset

get_ipython().system('rm -rf {InstanceImagesFolder}/.ipynb_checkpoints')
get_ipython().system('python "/content/Dreambooth-Stable-Diffusion/main.py"  --base configs/stable-diffusion/v1-finetune_unfrozen.yaml --seed {seed} -t  --actual_resume "model.ckpt"  --reg_data_root "{reg_data_root}"  -n "{project_name}"  --gpus 0,  --data_root "{InstanceImagesFolder}"  --max_training_steps {max_training_steps}  --class_word "{class_word}"  --token "{token}"  --no-test')


# ## Copy and name the checkpoint file

# In[ ]:


# Copy the checkpoint into our `trained_models` folder

directory_paths = get_ipython().getoutput('ls -d logs/*')
last_checkpoint_file = directory_paths[-1] + "/checkpoints/last.ckpt"
training_images = get_ipython().getoutput('find training_images/*')
date_string = get_ipython().getoutput('date +"%Y-%m-%dT%H-%M-%S"')
#file_name = date_string[-1] + "_" + project_name + "_" + str(len(training_images)) + "_training_images_" +  str(max_training_steps) + "_max_training_steps_" + token + "_token_" + class_word + "_class_word.ckpt"

#file_name = file_name.replace(" ", "_")
file_name = Session_Name + '.ckpt'


get_ipython().system('mv "{last_checkpoint_file}" "{SessionFolder}/{file_name}"')

