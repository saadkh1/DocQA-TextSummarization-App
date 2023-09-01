# Streamlit Document QA and Text Summarization App

This Streamlit application empowers users to effortlessly perform Document Question-Answering (QA) and Text Summarization tasks in their preferred language, English or German, with just a few simple steps.

## How to Use

### Language Selection

1. Select your preferred language (English or German) from the sidebar.

### Task Selection

2. Choose the task you want to perform - Document QA or Text Summarization.

### Document QA

#### 1. Upload Documents

- Click on the "Upload Files" section in the sidebar.
- Upload various types of documents including (PDFs, Markdown, plain text, and DOCX files).

#### 2. OCR for Images

- Click on the "OCR for Images" section in the sidebar.
- Conveniently upload images (PNG, JPG, JPEG) for optical character recognition (OCR).

#### 3. Upload Audio Files and Transcribe

- Click on the "Upload Audio Files and Transcribe" section in the sidebar.
- Effortlessly upload audio files (MP3, WAV) for automatic transcription.

#### 4. Import HTML

- Click on the "Import HTML" section in the sidebar.
- Simply enter URLs to import HTML content from websites.

#### 5. Transcribe YouTube Video

- Click on the "YouTube Video" section in the sidebar.
- Enter a YouTube video URL for transcription.

#### 6. Create Vector Database

- Click on the "Create Vector Database" section in the sidebar to create a database from uploaded documents.

#### 7. Remove All Files

- Click on the "Remove Files" section in the sidebar to remove all files in the data directory.

#### Chat with Chatbot

- Engage with a chatbot that can provide answers to questions based on the uploaded documents.

### Text Summarization

- In the Text Summarization task, simply enter text in the provided text area.
- Click the "Summarize" button to generate a concise summarization of the input text.

### Document QA Example:

Here's an example of the Document Question-Answering task in action:
![Document QA Example](https://github.com/saadkh1/DocQA-TextSummarization-App/blob/main/images/qa.png)

### Text Summarization Example:

And here's an example of the Text Summarization task in action:
![Text Summarization Example](https://github.com/saadkh1/DocQA-TextSummarization-App/blob/main/images/summarization.png)

## Installation and Running Locally

To use this Streamlit application, follow these steps:

1. **Clone the repository and navigate to the project directory:**

   ```bash
   git clone https://github.com/saadkh1/DocQA-TextSummarization-App.git
   ```
   ```bash
   cd DocQA-TextSummarization-App
   ```

2. **Install the required packages from the requirements.txt file:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the necessary language models and embeddings by running the models.sh script:**

    ```bash
    sh models.sh
    ```

4. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```
5. **Open this URL in your browser:** http://localhost:8501/


## Using Docker  

Alternatively, you can use Docker to run the application in a container. Make sure you have Docker installed on your system. Follow these steps:

1. **Clone the repository and navigate to the project directory:**

   ```bash
   git clone https://github.com/saadkh1/DocQA-TextSummarization-App.git
   ```
   ```bash
   cd DocQA-TextSummarization-App
   ```

2. **Build the Docker image:**

    ```bash
    docker build -t qa-summrize-app:1.0 .
    ```

3. **Run the Docker container:**

    ```bash
    docker run -p 8501:8501 qa-summrize-app:1.0
    ```

4. **Open this URL in your browser:** http://localhost:8501/

## Using Google Colab

If you prefer to use Google Colab, you can run the app using the provided app.ipynb notebook:

1. **Open the app.ipynb notebook in Google Colab:**

2. **Run all the cells in the notebook.**

The notebook will start the Streamlit app and expose it using ngrok. Follow the instructions in the notebook to access the app URL.
