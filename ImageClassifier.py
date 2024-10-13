#run first into google colab environment to prepare the environment, select GPU to process faster
!pip install --upgrade diffusers[torch]
!pip install transformers

#create image generation pipeline

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

###########################################################################
### generate images ###
import random
import matplotlib.pyplot as plt
import os

os.makedirs('/content/faces/happy', exist_ok=True)
os.makedirs('/content/faces/sad', exist_ok=True)
os.makedirs('/content/faces/angry', exist_ok=True)
os.makedirs('/content/faces/suprised', exist_ok=True)

ethnicities = ['a latino', 'a white', 'a black', 'a middle eastern', 'an indian', 'an asian']

genders = ['male', 'female']

emotion_prompts = {'happy': 'smiling, big smile, laugh',
                   'sad': 'frowning, sad face expression, crying, depressed, emotional, teary eyes',
                   'surprised': 'surprised, opened mouth, raised eyebrow',
                   'angry': 'angry, flared nostrils, tense jaws and lips, furrowed brows'}

for j in range(3):

  for emotion in emotion_prompts.keys():
    emotion_prompt = emotion_prompts[emotion]
    ethnicity = random.choice(ethnicities)
    gender = random.choice(genders)

    #print(emotion, ethnicity, gender)

    prompt = 'Medium-shot portrait of {} {}, {}, front view, looking at the camera, color photography, '.format(ethnicity, gender, emotion_prompt) + \
            'photorealistic, hyperrealistic, realistic, incredibly detailed, crisp focus, digital art, depth of field, 50mm, 8k'
    negative_prompt = '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ' + \
                      '((grayscale)) Low Quality, Worst Quality, plastic, fake, disfigured, deformed, blurry, bad anatomy, blurred, watermark, grainy, signature'

    img = pipeline(prompt, negative_prompt=negative_prompt).images[0]
    img.save('/content/faces/{}.png'.format(emotion, str(j).zfill(4)))
    #plt.imshow(img)
    #plt.show()


    #######################################################################
#first
!zip -r faces.zip /content/faces

#second
from google.colab import drive

drive.mount("/content/gdrive", force_remount=True)
#third 
!scp '/content/faces.zip' '/content/gdrive/My Drive/SyntheticFaceGeneration/faces.zip'





