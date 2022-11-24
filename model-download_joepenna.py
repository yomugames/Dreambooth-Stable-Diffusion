
# BUILD ENV
get_ipython().system('pip install omegaconf')
get_ipython().system('pip install einops')
get_ipython().system('pip install pytorch-lightning==1.6.5')
get_ipython().system('pip install test-tube')
get_ipython().system('pip install transformers')
get_ipython().system('pip install kornia')
get_ipython().system('pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers')
get_ipython().system('pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip')
get_ipython().system('pip install setuptools==59.5.0')
get_ipython().system('pip install pillow==9.0.1')
get_ipython().system('pip install torchmetrics==0.6.0')
get_ipython().system('pip install -e .')
get_ipython().system('pip install protobuf==3.20.1')
get_ipython().system('pip install gdown')
get_ipython().system('pip install -qq diffusers["training"]==0.3.0 transformers ftfy')
get_ipython().system('pip install -qq "ipywidgets>=7,<8"')
get_ipython().system('pip install huggingface_hub')
get_ipython().system('pip install ipywidgets==7.7.1')
get_ipython().system('pip install captionizer==1.0.1')


# In[ ]:

hf_token=input("Insert your huggingface token :")

# Hugging Face Login
from huggingface_hub import notebook_login

notebook_login()


# In[ ]:


# Download the 1.5 sd model
from IPython.display import clear_output

from huggingface_hub import hf_hub_download

if False:
  downloaded_model_path = hf_hub_download(
     repo_id="runwayml/stable-diffusion-v1-5",
     filename="v1-5-pruned.ckpt",
     token=hf_token
  )

  # Move the sd-v1-5.ckpt to the root of this directory as "model.ckpt"
  actual_locations_of_model_blob = get_ipython().getoutput('readlink -f {downloaded_model_path}')
  get_ipython().system('mv {actual_locations_of_model_blob[-1]} model.ckpt')
  clear_output()
  print("✅ model.ckpt successfully downloaded")



# # Regularization Images (Skip this section if you are uploading your own or using the provided images)

# Training teaches your new model both your token **but** re-trains your class simultaneously.
# 
# From cursory testing, it does not seem like reg images affect the model too much. However, they do affect your class greatly, which will in turn affect your generations.
# 
# You can either generate your images here, or use the repos below to quickly download 1500 images.

# In[ ]:


# GENERATE 200 images - Optional
self_generated_files_prompt = "person" #@param {type:"string"}
self_generated_files_count = 200 #@param {type:"integer"}

#get_ipython().system('python scripts/stable_txt2img.py  --seed 10  --ddim_eta 0.0  --n_samples 1  --n_iter {self_generated_files_count}  --scale 10.0  --ddim_steps 50  --ckpt model.ckpt  --prompt {self_generated_files_prompt}')

dataset=self_generated_files_prompt

#get_ipython().system('mkdir -p regularization_images/{dataset}')
#get_ipython().system('mv outputs/txt2img-samples/*.png regularization_images/{dataset}')


# In[ ]:


# Zip up the files for downloading and reuse.
# Download this file locally so you can reuse during another training on this dataset
get_ipython().system('apt-get install -y zip')
#get_ipython().system('zip -r regularization_images.zip regularization_images/{dataset}')


# # Download pre-generated regularization images
# We've created the following image sets
# 
# `man_euler` - provided by Niko Pueringer (Corridor Digital) - euler @ 40 steps, CFG 7.5
# `man_unsplash` - pictures from various photographers
# `person_ddim`
# `woman_ddim` - provided by David Bielejeski - ddim @ 50 steps, CFG 10.0
# `person_ddim` is recommended

# In[1]:


#Download Regularization Images

dataset="person_ddim" #@param ["man_euler", "man_unsplash", "person_ddim", "woman_ddim", "blonde_woman"]
get_ipython().system('git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-{dataset}.git')

get_ipython().system('mkdir -p regularization_images/{dataset}')
get_ipython().system('mv -v Stable-Diffusion-Regularization-Images-{dataset}/{dataset}/*.* regularization_images/{dataset}')
