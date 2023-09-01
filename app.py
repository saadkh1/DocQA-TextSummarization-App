import os
import re
from PIL import Image
import requests
import pytesseract
from pytube import YouTube
import whisper
import streamlit as st

from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
ENGLISH_MODEL_PATH = "llama-2-13b-chat.ggmlv3.q4_0.bin"
GERMAN_MODEL_PATH ="llama-2-13b-german-assistant-v2.ggmlv3.q4_0.bin"
MODEL_EMBEDDING_PATH = "all-MiniLM-L6-v2"
DATA_DIR = "data"

class DocumentQAApp:
    def __init__(self):
        self.selected_language = "English"  # Default to English
        self.llm = self.load_model()
        self.embeddings = self.create_huggingface_embeddings()
        self.run_chatbot = True  # Checkbox for chatbot (default: checked)
        self.run_summarizer = False  # Checkbox for document summarization (default: unchecked)

    # Function to save uploaded files
    def save_uploaded_file(self, uploaded_file):
        """
        Save an uploaded file to a specified directory.

        Args:
            uploaded_file (FileUploader): The uploaded file to be saved.
        """
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.sidebar.success(f"File '{uploaded_file.name}' saved to {DATA_DIR}")

    # Function to save an HTML file from a URL
    def save_html_from_url(self, url, data_dir):
        """
        Save an HTML file from a given URL to a specified directory.

        Args:
            url (str): The URL to the HTML file.
            data_dir (str): The directory where the HTML file should be saved.
        """
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Extract the file name from the URL
                file_name = url.split("/")[-1]
                file_name = file_name + ".html"  # Add .html extension
                file_path = os.path.join(data_dir, file_name)
                
                # Save the HTML content to the local file
                with open(file_path, "w", encoding="utf-8") as f:  # Use text mode 'w'
                    f.write(response.text)  # Write the HTML content as text

                return file_path
            else:
                st.sidebar.error(f"Failed to fetch HTML content from {url}. Status code: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"An error occurred while fetching the HTML content: {str(e)}")

    # Function to perform OCR on uploaded files and save as text
    def ocr_and_save_text(self, input_file, output_folder=DATA_DIR):
        """
        Perform OCR on scanned PDFs or images and save the extracted text as a .txt file.

        Args:
            input_file (str): The path to the input PDF or image file.
            output_folder (str): The folder where the output .txt file will be saved. Default is "data".

        Returns:
            str: The path to the saved .txt file.
        """
        # Check if the output folder exists; if not, create it.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Extract the title of the document from the input file name.
        title = os.path.splitext(os.path.basename(input_file))[0]

        # Perform OCR using pytesseract.
        try:
            if input_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img = Image.open(input_file)
                text = pytesseract.image_to_string(img)
            elif input_file.lower().endswith('.pdf'):
                text = pytesseract.image_to_string(Image.open(input_file), lang='eng', config='--psm 6')
            else:
                return None  # Unsupported file format.

            # Create the output .txt file path.
            txt_file_path = os.path.join(output_folder, f'{title}.txt')

            # Save the extracted text to the .txt file.
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

            # Remove the input image file.
            os.remove(input_file)
            
            return txt_file_path
        except Exception as e:
            print(f"An error occurred during OCR: {str(e)}")
            return None

    # Function to sanitize a string for use as a filename
    def sanitize_filename(self, filename):
        """ Remove characters that are not allowed in filenames"""
        return re.sub(r'[\/:*?"<>|]', '_', filename)

    # Function to transcribe audio file
    def transcribe_audio_file(self, audio_file):
        """
        Transcribe an audio file to text and save the transcription as a .txt file.

        Args:
            audio_file (FileUploader): The uploaded audio file to be transcribed.

        Returns:
            str: The path to the saved .txt file containing the transcription.
                Returns an empty string in case of an error.
        """
        try:
            model = whisper.load_model("base")
            audio_title = audio_file.name
            safe_audio_title = self.sanitize_filename(audio_title)
            
            # Save the uploaded audio file
            audio_path = os.path.join(DATA_DIR, audio_title)
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())
            
            # Transcribe the audio file
            transcription = model.transcribe(audio_path)
            
            # Create the output .txt file path
            txt_file_path = os.path.join(DATA_DIR, f"{safe_audio_title}.txt")
            
            # Save the transcription to the .txt file
            with open(txt_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(transcription["text"])
            
            # Remove the uploaded audio file
            os.remove(audio_path)
            
            return txt_file_path
        except Exception as e:
            return str(e)

    # Fonction pour transcrire une vidéo YouTube
    def transcribe_youtube_video(self, video_url):
        """
        Transcrit une vidéo YouTube en texte.

        Args:
            video_url (str): L'URL de la vidéo YouTube à transcrire.

        Returns:
            str: Le chemin du fichier texte contenant la transcription.
                 Retourne une chaîne vide en cas d'erreur.

        Raises:
            Exception: En cas d'erreur lors du téléchargement ou de la transcription.
        """
        try:
            model = whisper.load_model("base")
            yt = YouTube(video_url)
            video_title = yt.title
            safe_video_title = self.sanitize_filename(video_title)
            yt.streams.filter(only_audio=True).first().download(filename="audio.mp3")
            transcription = model.transcribe("audio.mp3")
            output_file = os.path.join(DATA_DIR, f"{video_title}.txt")
            with open(output_file, "w", encoding="utf-8") as text_file:
                text_file.write(transcription["text"])
            os.remove("audio.mp3")
            return output_file
        except Exception as e:
            return str(e)

    # Function to create a vector database
    def create_vector_database(self, data_dir):
        """
        Create a vector database from documents in a specified directory.

        Args:
            data_dir (str): The directory containing documents to be indexed.

        Returns:
            FAISS: An instance of FAISS (Fast Approximate Nearest Neighbors Index).
        """
        # Load various types of documents from the specified directory
        loaders = [
            DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(data_dir, glob="*.md", loader_cls=UnstructuredMarkdownLoader),
            DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader),
            DirectoryLoader(data_dir, glob="*.docx", loader_cls=Docx2txtLoader),
            DirectoryLoader(data_dir, glob="*.html", loader_cls=UnstructuredHTMLLoader),
        ]

        # Load and split documents into chunks for indexing
        loaded_documents = [doc for loader in loaders for doc in loader.load()]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunked_documents = text_splitter.split_documents(loaded_documents)

        # Build the vector database
        vector_database = FAISS.from_documents(
            documents=chunked_documents,
            embedding=self.embeddings,
        )

        # Save the vector database locally
        vector_database.save_local(DB_FAISS_PATH)

    # Function to remove all files in the DATA_DIR directory
    def remove_all_files(self, data_dir):
        """
        Remove all files in the specified directory.

        Args:
            data_dir (str): The directory from which files should be removed.
        """
        try:
            file_list = os.listdir(data_dir)
            for file_name in file_list:
                file_path = os.path.join(data_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            st.sidebar.success(f"All files in {data_dir} have been removed.")
        except Exception as e:
            st.sidebar.error(f"An error occurred while removing files: {str(e)}")

    # Function to load the language model
    def load_model(self, max_new_tokens=1000, temperature=0.7, n_ctx=2048):
        """
        Load a language model for generating responses in a conversation based on selected language.

        Args:
            model_path (str): The path to the language model file.
            max_new_tokens (int): The maximum number of tokens in generated responses.
            temperature (float): The temperature parameter for response generation.
            n_ctx (int): The context window size for the model.

        Returns:
            LlamaCpp: An instance of LlamaCpp, a language model for conversation.
        """
        if self.selected_language == "English":
            model_path = ENGLISH_MODEL_PATH
        elif self.selected_language == "German":
            model_path = GERMAN_MODEL_PATH
        else:
            raise ValueError("Invalid language selection")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model file found at {model_path}")

        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=model_path,
            n_ctx=n_ctx,
            max_tokens=max_new_tokens,
            temperature=temperature,
            callback_manager=callback_manager,
            verbose=True,
        )

        return llm

    # Function to create HuggingFace embeddings
    def create_huggingface_embeddings(self, model_name=MODEL_EMBEDDING_PATH):
        """
        Create HuggingFace embeddings for a given model name.

        Args:
            model_name (str): The name of the HuggingFace model.

        Returns:
            HuggingFaceEmbeddings: An instance of HuggingFaceEmbeddings.
        """
        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs = {'normalize_embeddings': False},
            )
        except Exception as e:
            raise Exception(f"Failed to load embeddings with model name {model_name}: {str(e)}")

    # Function to create a QA bot
    def create_qa_bot(self):
        """
        Create a Question-Answering (QA) bot using the specified components and configurations.

        Returns:
        - chain (ConversationalRetrievalChain): A configured QA bot instance ready for use.
        """
        vector_store = FAISS.load_local(folder_path=DB_FAISS_PATH, embeddings=self.embeddings)

        # Define templates for question-answering prompts based on language
        if self.selected_language == "English":
            template = """Use the following context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use a maximum of three sentences and keep the answer concise. 
            {context}
            Question: {question}
            Helpful Answer:"""
        elif self.selected_language == "German":
            template = """Verwenden Sie den folgenden Kontext, um die Frage am Ende zu beantworten. 
            Wenn Sie die Antwort nicht wissen, geben Sie einfach an, dass Sie es nicht wissen, und versuchen Sie nicht, eine Antwort zu erfinden. 
            Verwenden Sie maximal drei Sätze und halten Sie die Antwort prägnant. 
            {context}
            Frage: {question}
            Hilfreiche Antwort:"""
        else:
            raise ValueError("Invalid language selection")

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Initialize a conversation buffer memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Create a ConversationalRetrievalChain for handling conversations
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            # return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return chain

    # Function to handle conversation chat
    def conversation_chat(self, query):
        """
        Handle a conversation query and generate a response.

        Args:
            query (str): The user's query.

        Returns:
            str: The generated response.
        """
        chain = self.create_qa_bot()
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    # Initialize session state
    def initialize_session_state(self):
        """
        Initialize session-specific variables.
        """
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []

    # Display chat history and handle user input
    def display_chat_history(self):
        """
        Display the chat history and handle user input for the Streamlit app.
        """
        reply_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="Ask Here !", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = self.conversation_chat(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

    def generate_summarization(self, txt):
        """
        Generate a summarization for the given text.

        Args:
            txt (str): The input text to be summarized.

        Returns:
            str: The generated summarization.
        """
        # Split text
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(txt)
        
        # Create multiple documents from text
        docs = [Document(page_content=t) for t in texts]
        
        # Text summarization
        chain = load_summarize_chain(self.llm, chain_type='map_reduce')
        summarized_texts = chain.run(docs)  # Assumes chain.run returns a list of summarized texts
        
        return summarized_texts

    def display_summarization_results(self):
        """
        Display the summarization results.
        """

        # Text input
        txt_input = st.text_area("Enter the text to summarize:", "", height=200)

        # Form to accept user's text input for summarization
        with st.form("summarize_form", clear_on_submit=True):
            submitted = st.form_submit_button("Summarize")

            if submitted:
                # Generate the summarization
                response = self.generate_summarization(txt_input)
                st.subheader("Summarized Text")
                st.write(response)
    # Streamlit app setup
    def run(self):
        """
        Streamlit application entry point.
        """

        # Language selection
        st.sidebar.header("Select Language / Sprache auswählen")
        self.selected_language = st.sidebar.selectbox("Choose a language / Wählen Sie eine Sprache aus:", ["English", "German"])

        # Change UI text based on selected language
        if self.selected_language == "English":
            ui_texts = {
                "title_document_qa_bot": "Document QA Bot",
                "title_text_summarization": "Text Summarization",
                "upload_section": "Upload Documents",
                "ocr_section": "OCR for Images",
                "audio_section": "Upload Audio Files and Transcribe",
                "html_section": "Import HTML",
                "youtube_section": "YouTube Video",
                "youtube_url": "Enter YouTube Video URL",
                "db_section": "Create Vector Database",
                "remove_section": "Remove Files",
                "task_section": "Choose Task",
                "task_option_chatbot": "Chatbot",
                "task_option_summarization": "Summarization",
                "transcribe_button": "Transcribe Video",
                "upload_files_label": "Upload PDF, MD, TXT, or DOCX files",
                "upload_images_label": "Upload image files (PNG, JPG, JPEG)",
                "convert_html_button_label": "Convert HTML",
                "create_db_button_label": "Create Database",
                "remove_files_button_label": "Remove ALL Files",
            }
        elif self.selected_language == "German":
            ui_texts = {
                "title_document_qa_bot": "Dokumenten-Frage-Antwort-Bot",
                "title_text_summarization": "Textzusammenfassung",
                "upload_section": "Dokumente hochladen",
                "ocr_section": "OCR für Bilder",
                "audio_section": "Audiodateien hochladen und transkribieren",
                "html_section": "HTML importieren",
                "youtube_section": "YouTube-Video",
                "youtube_url": "Geben Sie die YouTube-Video-URL ein",
                "db_section": "Vektordatenbank erstellen",
                "remove_section": "Dateien entfernen",
                "task_section": "Aufgabe auswählen",
                "task_option_chatbot": "Chatbot",
                "task_option_summarization": "Zusammenfassung",
                "transcribe_button": "Video transkribieren",
                "upload_files_label": "PDF-, MD-, TXT- oder DOCX-Dateien hochladen",
                "upload_images_label": "Bilddateien hochladen (PNG, JPG, JPEG)",
                "convert_html_button_label": "HTML konvertieren",
                "create_db_button_label": "Datenbank erstellen",
                "remove_files_button_label": "Alle Dateien entfernen",
            }
        else:
            raise ValueError("Invalid language selection")

        # Choose Task
        st.sidebar.header(ui_texts["task_section"])
        self.selected_task = st.sidebar.radio("Select a task / Wählen Sie eine Aufgabe aus:", [ui_texts["task_option_chatbot"], ui_texts["task_option_summarization"]])

        if self.selected_task == ui_texts["task_option_chatbot"]:
            
            st.title(ui_texts["title_document_qa_bot"])

            # Section 1: Upload Documents
            st.sidebar.header(ui_texts["upload_section"])
            uploaded_files = st.sidebar.file_uploader(ui_texts["upload_files_label"], accept_multiple_files=True)

            if uploaded_files:
                for uploaded_file in uploaded_files:
                    self.save_uploaded_file(uploaded_file)
            
            # Section 2: OCR for Images
            st.sidebar.header(ui_texts["ocr_section"])
            uploaded_images = st.sidebar.file_uploader(ui_texts["upload_images_label"], accept_multiple_files=True)

            if uploaded_images:
                for uploaded_image in uploaded_images:
                    self.save_uploaded_file(uploaded_image)
                    if uploaded_image.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        ocr_result = self.ocr_and_save_text(os.path.join(DATA_DIR, uploaded_image.name))
                        if ocr_result:
                            st.sidebar.success(f"OCR performed on '{uploaded_image.name}' and text saved as '{os.path.basename(ocr_result)}'")

            # Section 3: Upload Audio Files and Transcribe
            st.sidebar.header(ui_texts["audio_section"])
            uploaded_audio_files = st.sidebar.file_uploader("Upload audio files (MP3, WAV)", accept_multiple_files=True)

            if uploaded_audio_files:
                for uploaded_audio_file in uploaded_audio_files:
                    # Check if the uploaded file is an audio file
                    if uploaded_audio_file.name.lower().endswith(('.mp3', '.wav')):
                        # Transcribe and save the audio file
                        transcription_result = self.transcribe_audio_file(uploaded_audio_file)
                        if transcription_result:
                            st.sidebar.success(f"Audio transcription saved as {transcription_result}")

            # Section 4: Import HTML
            st.sidebar.header(ui_texts["html_section"])
            url_input = st.sidebar.text_input("Enter URL(s) for HTML documents (separated by commas)")
            
            if st.sidebar.button(ui_texts["convert_html_button_label"]) and url_input:
                urls = url_input.split(",")
                for url in urls:
                    self.save_html_from_url(url.strip(), DATA_DIR)
                    st.sidebar.success(f"HTML content from {url} saved locally.")

            # Section 5: Transcribe YouTube Video
            st.sidebar.header(ui_texts["youtube_section"])
            youtube_url = st.sidebar.text_input("Enter YouTube Video URL")

            if st.sidebar.button(ui_texts["transcribe_button"]) and youtube_url:
                transcribed_file = self.transcribe_youtube_video(youtube_url)
                if transcribed_file:
                    st.sidebar.success(f"Transcription saved as {transcribed_file}")

            # Section 6: Create Vector Database
            st.sidebar.header(ui_texts["db_section"])
            if st.sidebar.button(ui_texts["create_db_button_label"]):
                self.create_vector_database(DATA_DIR)
                st.sidebar.success("Vector database created.")

            # Section 7: Remove All Files
            st.sidebar.header(ui_texts["remove_section"])
            if st.sidebar.button(ui_texts["remove_files_button_label"]):
                self.remove_all_files(DATA_DIR)
                
            # Run the chatbot
            self.initialize_session_state()
            self.display_chat_history()
        
        elif self.selected_task == ui_texts["task_option_summarization"]:
            
            st.title(ui_texts["title_text_summarization"])

            # Run the summarization function
            self.display_summarization_results()


if __name__ == "__main__":
    app = DocumentQAApp()
    app.run()