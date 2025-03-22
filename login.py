import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth

st.set_page_config(page_title="Gemini Pro Q/A Project", layout="wide", initial_sidebar_state="expanded")

# Initialize Firebase if not already initialized
if "firebase_initialized" not in st.session_state:
    cred = credentials.Certificate("./test-firestore-streamlit-c7e99-firebase-adminsdk-lm9jd-d02aeb541f.json")
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    
    st.session_state["firebase_initialized"] = True

# Add authentication state to session
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "About Us"
if 'selected_feature' not in st.session_state:
    st.session_state.selected_feature = "Select a Feature"



# Authentication functions
def login_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        st.session_state.authenticated = True
        st.session_state.user_email = email
        return user
    except auth.UserNotFoundError:
        st.error("User not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def register_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        st.success(f"User {email} successfully registered.")
        return user
    except Exception as e:
        st.error(f"Error registering: {e}")
        return None

def logout_user():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.selected_feature = "Select a Feature"
    st.rerun()

# Apply common styles [styles remain the same as in your original code]
st.markdown("""
    <style>
    /* [Your existing styles here] */
    </style>
""", unsafe_allow_html=True)

# Main app logic


if not st.session_state.authenticated:
    # Login/Register Interface
    st.title("Welcome to Saathi📚")
    menu = ["Login", "Sign Up"]
    choice = st.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login to Your Account")
        with st.form("login_form", clear_on_submit=True):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label="Login")
            
            if submit_button:
                user = login_user(email, password)
                if user:
                    st.success(f"Logged in as {email}")
                    st.rerun()

    elif choice == "Sign Up":
        st.subheader("Create a New Account")
        with st.form("signup_form", clear_on_submit=True):
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label="Sign Up")
            
            if submit_button:
                user = register_user(new_email, new_password)
                if user:
                    st.success("You have successfully created an account.")

else:
    # Navigation after login
    st.sidebar.title("Navigation")
    
    # Define About Us content function
    def show_about_content():
        # Apply custom CSS with !important tags and proper selectors
        st.markdown("""
            <style>
                /* Base styles */
                [data-testid="stAppViewContainer"] {
                    background-color: #1e1e1e !important;
                }
                
                .stApp {
                    background-color: #1e1e1e !important;
                }
                
                /* Sidebar styling */
                [data-testid="stSidebar"] {
                    background-color: #2c2c2c !important;
                    padding: 1rem !important;
                }
                
                /* Main content area styles */
                .main-content {
                    padding: 2rem !important;
                    color: #ffffff !important;
                }
                
                /* About section styles */
                .about-section {
                    background-color: #2c2c2c !important;
                    padding: 2rem !important;
                    border-radius: 15px !important;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
                    margin: 1rem 0 !important;
                    color: #ffffff !important;
                }
                
                .about-title {
                    font-size: 2.5rem !important;
                    color: #ffffff !important;
                    font-weight: 700 !important;
                    text-align: center !important;
                    margin-bottom: 1.5rem !important;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3) !important;
                }
                
                .feature-list {
                    font-size: 1.2rem !important;
                    line-height: 2 !important;
                    margin: 1rem 0 !important;
                    color: #ffffff !important;
                }
                
                .disclaimer {
                    background-color: #E74C3C !important;
                    color: #ffffff !important;
                    padding: 1rem !important;
                    border-radius: 10px !important;
                    font-size: 1.2rem !important;
                    font-weight: bold !important;
                    text-align: center !important;
                    margin: 1rem 0 !important;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
                }
                
                /* Image styles */
                .about-image {
                    border-radius: 15px !important;
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5) !important;
                    transition: transform 0.3s ease !important;
                }
                
                .about-image:hover {
                    transform: scale(1.05) !important;
                }
                
                /* Footer styles */
                .footer {
                    text-align: center !important;
                    color: #aaaaaa !important;
                    margin-top: 3rem !important;
                    padding: 1rem !important;
                    font-style: italic !important;
                }
                
                /* Feature description styles */
                .feature-description {
                    font-size: 1.1rem !important;
                    color: #ffffff !important;
                    margin-bottom: 1rem !important;
                    line-height: 1.6 !important;
                }
                
               
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("./1000_F_643686570_jlFaFaXfYNQKfvdnOAAYbY9E4AkUPWDb-removebg-preview.png", 
                    use_column_width=True,
                    caption="")
            st.markdown("<div class='disclaimer'>SAATHI IS BY YOU, WITH YOU AND FOR YOU! 🎯</div>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class='about-section'>
                    <h1 class='about-title'>Saathi 🎓</h1>
                    <div class='feature-description'>
                        Saathi is designed to provide personalized learning experiences, helping students to grasp concepts at their own pace.
                    </div>
                    <div class='feature-list'>
                        <strong>FEATURES:</strong><br>
                        💡 TO-DO LIST : Manage tasks efficiently. <br>
                        💡 ROADMAP : Strategic plans for key milestones. <br>
                        💡 CHATBOT : Instant academic support and guidance. <br>
                        💡 PDF ANALYSER : Ask questions from PDF files. <br>
                        💡 NOTES GENERATOR : Extract key info from PDFs. <br>
                        💡 QUIZ GENERATOR : Auto create quizzes from content. <br>
                        💡 CALCULATOR : Perform advance calculations. <br>
                        💡 STICKY NOTES : Store important info. <br>
                    </div>
                    <div class='disclaimer'>
                        <strong>Disclaimer:</strong> Saathi can make mistakes; please use it wisely!
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Session state initialization
    if 'show_about' not in st.session_state:
        st.session_state.show_about = True
        st.session_state.show_other_content = False

    # Sidebar navigation
    if st.sidebar.button("About Us"):
        st.session_state.show_about = True
        st.session_state.show_other_content = False
        st.session_state.selected_feature = "Select a Feature"
        st.rerun()

    # Feature selection
    features = ["Select a Feature", "To-Do List", "Roadmap Generator", "SAATHI Chatbot","Notes Generator", 
               "PDF Analyzer",  "Quiz Generator", "Calculator", "StickyNotes"]
    
    if st.session_state.selected_feature == "Select a Feature":
        displayed_features = features
    else:
        displayed_features = [feature for feature in features if feature != "Select a Feature"]
    
    page = st.sidebar.selectbox(
        "Choose a page",
        displayed_features,
        index=displayed_features.index(st.session_state.selected_feature)
    )
    
    # Update selection and state
    if page != st.session_state.selected_feature:
        st.session_state.selected_feature = page
        st.session_state.show_about = False
        st.session_state.show_other_content = True
        st.rerun()

    # Logout button
    if st.sidebar.button("Logout"):
        logout_user()
    
    # Content display logic
    if st.session_state.show_about:
        show_about_content()
    elif page != "Select a Feature":
        if page == "Quiz Generator":
            import quiz
            quiz.run_quiz()
        elif page == "Roadmap Generator":
            import roadmap
            roadmap.run_roadmap()
        elif page == "StickyNotes":
            import stickynotes
            stickynotes.run_stickynotes()
        elif page == "SAATHI Chatbot":
            import chatbot
            chatbot.run_chatbot()
        elif page == "Notes Generator":
            import notes
            notes.run_notes()
        elif page == "To-Do List":
            import todo
            todo.run_todo()
        elif page == "PDF Analyzer":
            import pdf_analyser
            pdf_analyser.run_pdf_analyser()
        elif page == "Calculator":
            import calculator
            calculator.run_calculator()

    # Footer
    st.markdown("<div class='footer'>© 2024 Saathi AI Tutor. All rights reserved.</div>", 
                unsafe_allow_html=True)