# import tkinter as tk


# from PIL import ImageTk


# import torch
# from torch import autocast
# from diffusers import StableDiffusionPipeline 

# modelid = "CompVis/stable-diffusion-v1-4"
# device = "cuda"
# pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token="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") 
# pipe.to(device)
# def generate(): 
#     with autocast(device): 
#         image = pipe("create a hacker in space", guidance_scale=8.5)
#         print(image)
    
#     image.save('generatedimage.png')
#     # img = ImageTk.PhotoImage(image)
# # generate()    
# print(torch.cuda.is_available())