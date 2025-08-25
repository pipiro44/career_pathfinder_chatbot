import os
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from dotenv import load_dotenv
import pickle
from typing import Dict, List, Optional
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv(r"C:\Users\Prospero!\Desktop\career chatbot\Chatbot\.env")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# Configuration
MODEL_NAME = "microsoft/DialoGPT-medium"
CHAT_HISTORY_DIR = "chat_history"
MAX_CONTEXT_LENGTH = 4000

# Configure Streamlit page
st.set_page_config(
    page_title="AI Career Pathfinder",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chat History Manager
class ChatHistoryManager:
    """Manages chat history persistence and loading"""
    
    def __init__(self):
        self.ensure_history_dir()
    
    def ensure_history_dir(self):
        """Ensure the chat history directory exists"""
        if not os.path.exists(CHAT_HISTORY_DIR):
            os.makedirs(CHAT_HISTORY_DIR)
    
    def save_history(self, messages: List[Dict]):
        """Save chat history to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.json"
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'messages': messages
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Failed to save chat history: {str(e)}")
    
    def load_latest_history(self) -> List[Dict]:
        """Load the most recent chat history"""
        try:
            files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith('.json')]
            if not files:
                return []
            
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(CHAT_HISTORY_DIR, x)))
            filepath = os.path.join(CHAT_HISTORY_DIR, latest_file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('messages', [])
        except Exception as e:
            st.warning(f"Could not load chat history: {str(e)}")
            return []
    
    def clear_history(self):
        """Clear all chat history files"""
        try:
            files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith('.json')]
            for file in files:
                os.remove(os.path.join(CHAT_HISTORY_DIR, file))
            return True
        except Exception as e:
            st.error(f"Failed to clear history: {str(e)}")
            return False

# Initialize session state
def initialize_session_state():
    defaults = {
        'user_name': None,
        'chat_history': [],
        'current_step': 0,
        'user_responses': {},
        'prediction_complete': False,
        'is_typing': False,
        'last_user_input': None,
        'conversation': None,
        'history_store': {},
        'predicted_cluster': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Load the trained model for career prediction
@st.cache_resource
def load_model():
    try:
        with open(r"C:\Users\Prospero!\Desktop\career chatbot\Chatbot\best_model.pkl", 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file 'best_model.pkl' not found. Please ensure the model file is available.")
        return None

# RAG System for Career Information
class CareerRAG:
    def __init__(self):
        self.career_database = {
            "STEM": {
                "salary_range": "$60,000 - $150,000+",
                "growth_rate": "8% annually",
                "top_companies": ["Google", "Microsoft", "Apple", "Tesla", "SpaceX"],
                "skills_trends": ["AI/ML", "Cloud Computing", "Cybersecurity", "Data Science"],
                "education_trends": ["Online certifications", "Bootcamps", "Advanced degrees"]
            },
            "Healthcare": {
                "salary_range": "$50,000 - $200,000+",
                "growth_rate": "13% annually",
                "top_companies": ["Mayo Clinic", "Cleveland Clinic", "Johns Hopkins", "Stanford Health"],
                "skills_trends": ["Telemedicine", "Digital Health", "Patient Care Technology"],
                "education_trends": ["Online nursing programs", "Healthcare certifications"]
            },
            "Business": {
                "salary_range": "$45,000 - $200,000+",
                "growth_rate": "7% annually",
                "top_companies": ["McKinsey", "Bain", "BCG", "Deloitte", "PwC"],
                "skills_trends": ["Digital Marketing", "Data Analytics", "Remote Management"],
                "education_trends": ["Online MBA programs", "Business certifications"]
            },
            "Arts/Humanities": {
                "salary_range": "$30,000 - $100,000+",
                "growth_rate": "4% annually",
                "top_companies": ["Netflix", "Disney", "Adobe", "Creative Agencies"],
                "skills_trends": ["Digital Design", "Content Creation", "Social Media"],
                "education_trends": ["Online design courses", "Portfolio building"]
            },
            "Vocational": {
                "salary_range": "$40,000 - $80,000+",
                "growth_rate": "5% annually",
                "top_companies": ["Trade unions", "Construction firms", "Manufacturing companies"],
                "skills_trends": ["Green technology", "Automation", "Safety protocols"],
                "education_trends": ["Apprenticeships", "Technical certifications"]
            }
        }

    def search_career_info(self, career_cluster: str, query: str = None) -> Dict:
        base_info = self.career_database.get(career_cluster, {})
        if query:
            if "salary" in query.lower():
                return {
                    "type": "salary_info",
                    "data": base_info.get("salary_range", "Information not available"),
                    "source": "Bureau of Labor Statistics (simulated)",
                    "timestamp": datetime.now().strftime("%Y-%m-%d")
                }
            elif "growth" in query.lower():
                return {
                    "type": "growth_info",
                    "data": base_info.get("growth_rate", "Information not available"),
                    "source": "Employment Projections (simulated)",
                    "timestamp": datetime.now().strftime("%Y-%m-%d")
                }
            elif "companies" in query.lower():
                return {
                    "type": "companies_info",
                    "data": base_info.get("top_companies", []),
                    "source": "Industry Reports (simulated)",
                    "timestamp": datetime.now().strftime("%Y-%m-%d")
                }
        return {
            "type": "general_info",
            "data": base_info,
            "source": "Career Database",
            "timestamp": datetime.now().strftime("%Y-%m-%d")
        }

# Fallback conversation system
class FallbackConversation:
    def __init__(self, predicted_cluster):
        self.predicted_cluster = predicted_cluster or "General"
        self.conversation_history = []
        self.responses = {
            "greeting": [
                f"Hello! I'm here to help you with your {self.predicted_cluster} career path. What would you like to know?",
                f"Hi there! Ready to explore your {self.predicted_cluster} career options? Ask me anything!",
                f"Welcome! I'm excited to help you navigate your {self.predicted_cluster} career journey."
            ],
            "salary": {
                "STEM": "STEM careers typically offer excellent salaries ranging from $60,000 to $150,000+. Tech roles like software engineering ($80K-$200K), data science ($90K-$180K), and cybersecurity ($70K-$160K) are particularly well-compensated.",
                "Healthcare": "Healthcare professionals can earn between $50,000 to $200,000+. Nurses ($60K-$90K), physical therapists ($80K-$95K), and specialized doctors ($200K+) command good salaries.",
                "Business": "Business careers range from $45,000 to $200,000+. Management consulting ($100K-$300K), finance roles ($70K-$250K), and executive positions offer the highest compensation.",
                "Arts/Humanities": "Creative careers range from $30,000 to $100,000+. Digital media specialists ($45K-$85K), UX designers ($65K-$120K), and creative directors ($80K-$150K) are growing fields.",
                "Vocational": "Skilled trades offer $40,000 to $80,000+ with excellent job security. Electricians ($50K-$80K), plumbers ($45K-$75K), and HVAC technicians ($40K-$70K) have strong growth potential."
            },
            "skills": {
                "STEM": "Focus on: Programming (Python, Java, JavaScript), data analysis, cloud computing (AWS, Azure), cybersecurity fundamentals, problem-solving, mathematics, and continuous learning.",
                "Healthcare": "Develop: Patient care techniques, medical terminology, communication skills, attention to detail, empathy, teamwork, stress management, and technology proficiency.",
                "Business": "Build: Leadership and management, communication and presentation, analytical thinking, project management, strategic planning, digital marketing, and financial analysis.",
                "Arts/Humanities": "Cultivate: Creativity and innovation, communication and storytelling, cultural awareness, digital literacy, design software proficiency, and adaptability to new media trends.",
                "Vocational": "Master: Technical and hands-on skills specific to your trade, safety protocols and regulations, problem-solving, attention to detail, and customer service skills."
            },
            "education": {
                "STEM": "Consider: Computer science or engineering degrees (4-year), coding bootcamps (3-6 months), online certifications (Coursera, edX, Udacity), and professional certifications (AWS, Google Cloud).",
                "Healthcare": "Pursue: Medical degrees (4+ years), nursing programs (2-4 years), healthcare certifications, specialized training programs, and continuing education for licensing.",
                "Business": "Explore: Business administration degrees, MBA programs (2 years), professional certifications (PMP, Six Sigma), and leadership development programs.",
                "Arts/Humanities": "Try: Liberal arts degrees, portfolio development, creative workshops and masterclasses, digital skills training, and internships at creative agencies.",
                "Vocational": "Look into: Trade schools (6 months-2 years), apprenticeship programs (2-4 years), technical certifications, and hands-on training programs."
            },
            "default": [
                f"That's a great question about your {self.predicted_cluster} career path! Based on your assessment results, you're well-suited for this field. What specific aspect would you like to explore further?",
                f"I'd love to help you with that! Your {self.predicted_cluster} career cluster offers many exciting opportunities. Let me provide some insights.",
                f"Career decisions can be challenging, but you're on the right track with {self.predicted_cluster}. Let's explore this together!"
            ]
        }
    
    def invoke(self, inputs, config=None):
        user_input = inputs.get("input", "").lower()
        self.conversation_history.append(user_input)
        
        # Simple keyword matching for responses
        if any(word in user_input for word in ["salary", "pay", "money", "income", "earn", "wage"]):
            return self.responses["salary"].get(self.predicted_cluster, "Salary varies widely by role and location. Research specific positions in your area for accurate information.")
        
        elif any(word in user_input for word in ["skill", "ability", "competency", "talent"]):
            return self.responses["skills"].get(self.predicted_cluster, "Focus on developing both technical and soft skills relevant to your field.")
        
        elif any(word in user_input for word in ["education", "degree", "study", "learn", "school"]):
            return self.responses["education"].get(self.predicted_cluster, "Education requirements vary by field. Consider formal degrees, professional certifications, and continuous learning.")
        
        elif any(word in user_input for word in ["hello", "hi", "hey", "greeting"]):
            return np.random.choice(self.responses["greeting"])
        
        else:
            return np.random.choice(self.responses["default"]).replace("{predicted_cluster}", self.predicted_cluster)

# Function to get session history
def get_session_history(session_id: str) -> ChatMessageHistory:
    """Get or create a ChatMessageHistory for the given session_id"""
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]

# Initialize LangChain conversation
@st.cache_resource
def initialize_conversation(predicted_cluster: Optional[str] = None):
    if not HUGGINGFACEHUB_API_TOKEN:
        st.warning("HUGGINGFACEHUB_API_TOKEN not found. Using fallback system.")
        return FallbackConversation(predicted_cluster)
    
    try:
        model_options = [
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-medium", 
            "facebook/blenderbot-90M-distill"
        ]
        
        llm = None
        for model_name in model_options:
            try:
                llm = HuggingFaceEndpoint(
                    repo_id=model_name,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
                )
                
                # Test the model
                test_response = llm.invoke("Hello")
                if test_response and len(str(test_response).strip()) > 0:
                    st.success(f"Successfully loaded model: {model_name}")
                    break
                    
            except Exception as model_error:
                st.warning(f"Model {model_name} failed: {str(model_error)}")
                continue
        
        if llm is None:
            st.warning("All external models failed. Using fallback system.")
            return FallbackConversation(predicted_cluster)
            
        cluster_info = (
            f"The user's predicted career cluster is: {predicted_cluster}. Provide personalized advice."
            if predicted_cluster and predicted_cluster != "Undecided"
            else "The user is undecided about their career path. Help them explore options."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert AI Career Pathfinder assistant. Provide helpful, personalized career advice.

Your responses should be:
- Helpful, informative, and encouraging
- Personalized to the user's situation
- Brief and focused (1-2 paragraphs max)
- Actionable with concrete next steps

Cluster Info: {cluster_info}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        chain = prompt | llm
        
        conversation = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        return conversation
        
    except Exception as e:
        st.error(f"Failed to initialize conversation: {str(e)}")
        return FallbackConversation(predicted_cluster)

# Interactive Questions
def create_interactive_questions():
    model_data = load_model()
    if model_data is None:
        return None
        
    questions = {
        'subjects_excel_Biology': {
            'question': "Do you excel in Biology as a subject?",
            'options': ['Yes', 'No'],
            'help': "Select 'Yes' if Biology is one of your strongest subjects",
            'type': 'binary'
        },
        'social_impact_freq': {
            'question': "How often do you participate in community or social impact activities?",
            'options': ['Never', 'Rarely', 'Sometimes', 'Often', 'Daily'],
            'help': "Activities like volunteering, community service, or social causes",
            'type': 'frequency'
        },
        'career_motivation_High_salary': {
            'question': "Is earning a high salary important to you in your career choice?",
            'options': ['Yes', 'No'],
            'help': "Select 'Yes' if financial compensation is a key motivator for you",
            'type': 'binary'
        },
        'career_motivation_Job_stability': {
            'question': "Is job stability and security important to you in your career choice?",
            'options': ['Yes', 'No'],
            'help': "Select 'Yes' if having a stable, secure job is important to you",
            'type': 'binary'
        },
        'strongest_skills_Problem-solving': {
            'question': "Do you consider problem-solving to be one of your strongest skills?",
            'options': ['Yes', 'No'],
            'help': "Select 'Yes' if you're good at analyzing problems and finding solutions",
            'type': 'binary'
        }
    }
    return questions

def preprocess_user_input(responses, model_data):
    try:
        selected_features = model_data['selected_features']
        user_data = pd.DataFrame([responses])
        frequency_mapping = {'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Daily': 5}
        
        for col in user_data.columns:
            if col == 'social_impact_freq':
                user_data[col] = user_data[col].map(frequency_mapping)
        
        user_data_selected = user_data[selected_features]
        return user_data_selected
    except Exception as e:
        st.error(f"Error preprocessing user input: {str(e)}")
        default_data = pd.DataFrame([[0] * len(selected_features)], columns=selected_features)
        return default_data

def predict_career_cluster(user_data, model_data):
    try:
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        prediction = model.predict(user_data)[0]
        target_column = 'career_cluster'
        predicted_cluster = label_encoders[target_column].inverse_transform([prediction])[0]
        
        if predicted_cluster == "Undecided":
            return predicted_cluster, "Based on your answers, you're exploring various interests! 'Undecided' means your responses didn't point strongly to one field yet. This is normal and gives you flexibility to explore different paths."
        
        return predicted_cluster, None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return "Undecided", "Something went wrong with the prediction. Let's try again or explore some general career options!"

# Main Chat Interface
def main_chat_interface():
    st.title("AI Career Pathfinder")
    st.markdown("*Powered by AI and Machine Learning*")
    st.markdown("---")
    
    initialize_session_state()
    history_manager = ChatHistoryManager()
    career_rag = CareerRAG()
    model_data = load_model()
    
    if model_data is None:
        st.error("Unable to load model. Please check the model file path.")
        st.stop()
    
    questions = create_interactive_questions()
    if questions is None:
        st.stop()
    
    if st.session_state.conversation is None:
        st.session_state.conversation = initialize_conversation()
    
    # Sidebar
    with st.sidebar:
        st.header("User Profile")
        if st.session_state.user_name:
            st.success(f"Welcome, {st.session_state.user_name}!")
        else:
            st.info("Please enter your name to start")
        
        st.header("Assessment Progress")
        progress = len(st.session_state.user_responses) / len(questions)
        st.progress(progress)
        st.write(f"{len(st.session_state.user_responses)}/{len(questions)} questions completed")
        
        if st.session_state.prediction_complete:
            st.header("Your Career Profile")
            st.info(f"**Predicted Cluster:** {st.session_state.predicted_cluster}")
        
        st.header("Chat Controls")
        if st.button("Clear Chat History"):
            if history_manager.clear_history():
                st.session_state.chat_history = []
                st.session_state.history_store = {}
                st.success("Chat history cleared!")
                st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Career Conversation")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"Sent at: {message['timestamp']}")
        
        # User name input
        if not st.session_state.user_name:
            st.info("Welcome! Let's start by getting to know you.")
            user_name = st.text_input("What's your name?", placeholder="Enter your name here...")
            if st.button("Start Assessment", type="primary"):
                if user_name.strip():
                    st.session_state.user_name = user_name
                    welcome_message = f"Hello {user_name}! I'm your AI Career Pathfinder. I'll help you discover your ideal career path through a personalized assessment."
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': welcome_message,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    if "default" not in st.session_state.history_store:
                        st.session_state.history_store["default"] = ChatMessageHistory()
                    st.session_state.history_store["default"].add_ai_message(welcome_message)
                    history_manager.save_history(st.session_state.chat_history)
                    st.rerun()
                else:
                    st.error("Please enter your name to continue.")
        
        # Assessment questions
        elif not st.session_state.prediction_complete:
            st.info(f"Hi {st.session_state.user_name}! Let's assess your career preferences.")
            selected_features = model_data['selected_features']
            
            if st.session_state.current_step < len(selected_features):
                current_feature = selected_features[st.session_state.current_step]
                question_data = questions[current_feature]
                
                st.subheader(f"Question {st.session_state.current_step + 1} of {len(selected_features)}")
                st.write(f"**{question_data['question']}**")
                
                if question_data['type'] == 'binary':
                    response = st.radio(
                        "Your answer:",
                        options=question_data['options'],
                        help=question_data['help']
                    )
                else:
                    response = st.selectbox(
                        "Your answer:",
                        options=question_data['options'],
                        help=question_data['help']
                    )
                
                if st.button("Next Question", type="primary"):
                    if question_data['type'] == 'binary':
                        st.session_state.user_responses[current_feature] = 1 if response == 'Yes' else 0
                    else:
                        st.session_state.user_responses[current_feature] = response
                    
                    user_response = f"Q{st.session_state.current_step + 1}: {question_data['question']} - {response}"
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_response,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    if "default" not in st.session_state.history_store:
                        st.session_state.history_store["default"] = ChatMessageHistory()
                    st.session_state.history_store["default"].add_user_message(user_response)
                    
                    st.session_state.current_step += 1
                    
                    # Complete assessment
                    if st.session_state.current_step >= len(selected_features):
                        with st.spinner("Analyzing your responses..."):
                            user_data = preprocess_user_input(st.session_state.user_responses, model_data)
                            predicted_cluster, undecided_message = predict_career_cluster(user_data, model_data)
                            
                            if undecided_message:
                                message = undecided_message
                            else:
                                message = f"Analysis complete! Based on your responses, I predict you'd excel in the **{predicted_cluster}** career cluster. Let me provide you with detailed insights and recommendations."
                            
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': message,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            st.session_state.history_store["default"].add_ai_message(message)
                            st.session_state.prediction_complete = True
                            st.session_state.predicted_cluster = predicted_cluster
                            
                            # Reinitialize conversation with updated cluster
                            st.session_state.conversation = initialize_conversation(predicted_cluster)
                            history_manager.save_history(st.session_state.chat_history)
                            st.rerun()
        
        # Chat interface after assessment
        else:
            st.success("Assessment Complete! You can now ask me questions about your career path.")
            user_input = st.chat_input("Ask me about your career path, skills, education, or anything else...")
            
            if user_input and user_input != st.session_state.last_user_input and user_input.strip():
                st.session_state.last_user_input = user_input
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': timestamp
                })
                
                with st.chat_message("user"):
                    st.markdown(user_input)
                    st.caption(f"Sent at: {timestamp}")
                
                with st.chat_message("assistant"):
                    with st.spinner("AI is thinking..."):
                        try:
                            rag_info = career_rag.search_career_info(st.session_state.predicted_cluster, user_input)
                            
                            # Get AI response
                            try:
                                response = st.session_state.conversation.invoke(
                                    {"input": user_input},
                                    config={"configurable": {"session_id": "default"}}
                                )
                                
                                if hasattr(response, 'content'):
                                    ai_response = response.content
                                elif isinstance(response, dict):
                                    ai_response = response.get("text", response.get("content", str(response)))
                                else:
                                    ai_response = str(response)
                                
                                # Clean response
                                if ai_response.startswith("Human:") or ai_response.startswith("Assistant:"):
                                    ai_response = ai_response.split(":", 1)[-1].strip()
                                
                                if len(ai_response.strip()) < 10:
                                    raise ValueError("Response too short")
                                
                            except Exception as model_error:
                                st.info("Using offline mode")
                                fallback = FallbackConversation(st.session_state.predicted_cluster)
                                ai_response = fallback.invoke({"input": user_input})
                            
                            # Add RAG information
                            if rag_info['type'] != 'general_info':
                                if rag_info['type'] == 'salary_info':
                                    ai_response += f"\n\n**Current Salary Data:** {rag_info['data']}"
                                elif rag_info['type'] == 'growth_info':
                                    ai_response += f"\n\n**Growth Outlook:** {rag_info['data']}"
                                elif rag_info['type'] == 'companies_info':
                                    ai_response += f"\n\n**Top Employers:** {', '.join(rag_info['data'][:5])}"
                            
                            st.markdown(ai_response)
                            bot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': ai_response,
                                'timestamp': bot_timestamp
                            })
                            st.caption(f"Response at: {bot_timestamp}")
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            fallback = FallbackConversation(st.session_state.predicted_cluster)
                            ai_response = fallback.invoke({"input": user_input})
                            st.markdown(ai_response)
                            
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': ai_response,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                
                history_manager.save_history(st.session_state.chat_history)
                st.rerun()
    
    # Right column - Career insights
    with col2:
        if st.session_state.prediction_complete:
            st.subheader("Career Insights")
            career_info = career_rag.search_career_info(st.session_state.predicted_cluster)
            
            if career_info['data']:
                st.subheader("Career Overview")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Salary Range", career_info['data'].get('salary_range', 'N/A'))
                with col_b:
                    st.metric("Growth Rate", career_info['data'].get('growth_rate', 'N/A'))
                
                st.subheader("Top Companies")
                companies = career_info['data'].get('top_companies', [])
                for company in companies[:3]:
                    st.write(f"• {company}")
                
                st.subheader("Trending Skills")
                skills = career_info['data'].get('skills_trends', [])
                for skill in skills:
                    st.write(f"• {skill}")
                
                st.subheader("Quick Actions")
                if st.button("Get Education Path"):
                    education_info = career_rag.search_career_info(st.session_state.predicted_cluster, "education")
                    st.info(f"Education trends: {', '.join(education_info['data'].get('education_trends', ['Information not available']))}")
                
                if st.button("Salary Research"):
                    salary_info = career_rag.search_career_info(st.session_state.predicted_cluster, "salary")
                    st.info(f"Salary range: {salary_info['data']}")
        
        st.subheader("Chat Tips")
        st.write("""
        Try asking me about:
        • Salary expectations and ranges
        • Required skills and competencies
        • Education paths and requirements
        • Job growth trends and outlook
        • Company recommendations
        • Industry insights and trends
        
        Note: If external AI models are slow, the system will use fast offline responses.
        """)

if __name__ == "__main__":
    main_chat_interface()