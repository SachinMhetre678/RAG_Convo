from flask import Flask, render_template, request, jsonify, session, send_file
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import pickle
import uuid
import hashlib
import io
import traceback
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Global store for chat histories
chat_store = {}

# Store for vectorstore
vectorstore = None

# Ensure upload directory exists
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html', session_id=session['session_id'])

@app.route('/new_session', methods=['POST'])
def new_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    chat_store[session_id] = ChatMessageHistory()
    return jsonify({'success': True, 'session_id': session_id})

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    global vectorstore
    
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided in request'}), 400
        
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400

        # Create a list to store processed files
        processed_files = []
        file_hashes = []

        for file in files:
            if file and file.filename.endswith('.pdf'):
                # Create a BytesIO object to store the file content
                file_content = file.read()
                file_hash = hashlib.md5(file_content).hexdigest()
                file_hashes.append(file_hash)

                # Save the file temporarily
                temp_file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
                with open(temp_file_path, 'wb') as f:
                    f.write(file_content)

                # Process the PDF
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load_and_split()
                
                # Clean up the temporary file
                os.remove(temp_file_path)
                
                processed_files.extend(pages)

        if not processed_files:
            return jsonify({'error': 'No valid PDF files found'}), 400

        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(processed_files)

        # Create or update vectorstore
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        session['file_hashes'] = tuple(file_hashes)

        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(files)} PDF files'
        })

    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
    # Define the system prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

system_prompt = (
       "You are an assistant that answers questions strictly based on the content of uploaded PDFs. "
            "Follow these rules:\n"
            "1. If the question is unrelated to the PDFs, respond with: 'This question is not related to the uploaded documents.'\n"
            "2. If the PDFs contain relevant information, provide a concise response (maximum three sentences).\n"
            "3. If no relevant information is found, state: 'I don't know.'\n"
            "4. Do not speculate or provide information outside the scope of the uploaded documents."
    "\n\n"
    "{context}"
)

@app.route('/ask', methods=['POST'])
def ask_question():
    global vectorstore
    
    try:
        data = request.json
        user_input = data.get('question', '')
        api_key = data.get('api_key', '')
        session_id = data.get('session_id', session.get('session_id'))
        
        if not api_key:
            return jsonify({'error': 'Groq API key is required'}), 400
        
        if not user_input:
            return jsonify({'error': 'Question is required'}), 400
        
        if vectorstore is None:
            return jsonify({'error': 'Please upload PDF documents first'}), 400
            
        # Setup RAG chain
        llm = ChatGroq(groq_api_key=api_key, model_name="deepseek-r1-distill-llama-70b")
        retriever = vectorstore.as_retriever()
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        answer = response.get('answer', 'No answer generated')
        
        session_history = get_session_history(session_id)
        history_messages = [{"role": "human" if i % 2 == 0 else "ai", 
                            "content": msg.content} 
                            for i, msg in enumerate(session_history.messages)]
        
        return jsonify({
            'answer': answer,
            'history': history_messages
        })
        
    except Exception as e:
        print(f"Ask error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f"Server error: {str(e)}"}), 500

def get_session_history(session_id):
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)