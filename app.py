import streamlit as st

def main():
    st.set_page_config(page_title="PDF Chatting AI", page_icon=":books:", layout="centered")
    st.title("PDF Chatting AI")
    st.text_input("Upload a PDF file and start chatting with it!")
    with st.sidebar:
         st.subheader("Upload PDF")
         st.file_uploader("Choose a PDF file", type=["pdf"])    
         st.button("Process PDF")
         

if __name__ == "__main__":    main()
                
        
    
        # Here you can add code to process the PDF and enable chatting functionality