import requests
from bs4 import BeautifulSoup
import streamlit as st

from chatapp import get_text_chunks, get_vector_store, user_input

url = 'https://www.geeksforgeeks.org/'
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')

urls = []
for link in soup.find_all('a'):
    print(link.get('href'))


def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    user_question = st.text_input(
        "Ask a Question from the PDF Files uploaded .. ✍️📝")

    # if user_question:
    #     user_input(user_question)

    if user_question:
        user_input(user_question)
    with st.sidebar:

        st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 PDF File's Section")
        pdf_docs = st.text_input("enter website link")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                url = pdf_docs

                reqs = requests.get(url)
                soup = BeautifulSoup(reqs.text, 'html.parser')

                urls = []
                # for link in soup.get_text(separator='\n'):
                #     print(link, ":::::::::::::::::::::::::::::")

                # user friendly message.
                # print(pdf_docs, "====================")
                print(soup.body.text, "====================")
                raw_text = soup.body.text
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
