# import requests
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
# from transformers import ViltProcessor, ViltForQuestionAnswering
# import requests
# from PIL import Image
# from diffusers import StableDiffusionPipeline
# import torch
# from diffusers import StableDiffusionInpaintPipeline

from dotenv import load_dotenv

load_dotenv()
# os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def text_imag():


#     model_id = "runwayml/stable-diffusion-v1-5"
#     pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,revision="fp16")
#     pipe = pipe.to("cuda")

#     prompt = "iron vs superman"
#     image = pipe(prompt).images[0]

#     image.save("astronut coding.png")
# def pre():
#     pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     revision="fp16",
#     torch_dtype=torch.float16,
# )
#     prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
# #image and mask_image should be PIL images.
# #The mask structure is white for inpainting and black for keeping as is
#     image = pipe(prompt=prompt).images[0]
#     image.save("./yellow_cat_on_park_bench.png")
# def imag():
#     url = "https://images.unsplash.com/photo-1566438480900-0609be27a4be?q=80&w=1894&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
#     st.image(url,caption=url)
#     image = Image.open(requests.get(url, stream=True).raw)
#     text =  st.text_input("what color in this images")

#     processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
#     model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

#     # prepare inputs
#     encoding = processor(image, text, return_tensors="pt")

#     # forward pass
#     outputs = model(**encoding)
#     logits = outputs.logits
#     idx = logits.argmax(-1).item()
#     print("Predicted answer:", model.config.id2label[idx])
#     st.write(f"{model.config.id2label[idx]}")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print(f"{chunks},--------------------- get_text_chunks")
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=1)

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    user_question = st.text_input(
        "Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:

        st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):  # user friendly message.
                raw_text = get_pdf_text(pdf_docs)
                # get the pdf text
                text_chunks = get_text_chunks(raw_text)  # get the text chunks
                get_vector_store(text_chunks)  # create vector store
                st.success("Done")

        st.write("---")
        # st.write(get_pdf_text(pdf_docs))
        # st.image("img/gkj.jpg")
        # add this line to display the image
        st.write("AI App created by @ Jeet")
    # imag()
    # text_imag()
    # pre()
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.memory_allocated())

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/shahjeet23" target="_blank">Team Dracula</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
