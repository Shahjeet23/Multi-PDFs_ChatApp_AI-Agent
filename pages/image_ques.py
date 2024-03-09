import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# Use a pipeline as a high-level helper
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
from dotenv import load_dotenv
def imag(url):
    # url = "https://i.pinimg.com/564x/3d/cf/e1/3dcfe1b48c7c7fc873ca627a4ac0b729.jpg"
    st.image(url )
    image = Image.open(requests.get(url, stream=True).raw)
    text =  st.text_input("ask the question regarding the image" )

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
    st.write(f"{model.config.id2label[idx]}")
def main():
     st.title("projects")
   
     st.header("image to text (question  answering)")
     with st.sidebar:

        st.image("img/Robot.jpg")
        st.write("---")
        
        st.title("📁 Image url section")
        # pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        imag_url=st.text_input('Enter URL')
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): # user friendly message.
                 # create vector store
                st.success("Done")
        
        st.write("---")
        # st.image("img/gkj.jpg")
        st.write("AI App created by @ Jeet")
     if imag_url:
            imag(imag_url)
     else:
            imag("https://st3.depositphotos.com/1765681/13307/i/380/depositphotos_133078244-stock-photo-siberian-tiger-panthera-tigris-altaica.jpg")   

if __name__=="__main__":
    main()
   

   


