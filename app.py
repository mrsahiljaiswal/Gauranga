import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Add the utils directory to the Python path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from utils.tremor_analysis import TremorAnalyzer
from utils.tap_analysis import TapAnalyzer
from utils.data_handler import DataHandler
from utils.risk_calculator import RiskCalculator
from utils.visualization import VisualizationEngine

# Set page configuration
st.set_page_config(
    page_title="Parkinson's Disease Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"
    st.session_state.user_id = f"user_{int(time.time())}"
    st.session_state.test_in_progress = False
    st.session_state.landmarks_data = []
    st.session_state.test_start_time = None

# Initialize data handler
data_handler = DataHandler()
visualizer = VisualizationEngine()

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-moderate {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("🧠 PD Detection")
    st.markdown("---")
    
    # Navigation
    pages = {
        "Dashboard": "🏠 Dashboard",
        "Tremor Test": "📊 Tremor Test",
        "Tap Test": "👆 Tap Test",
        "Symptom Survey": "📝 Symptom Survey",
        "Results & History": "📈 Results & History",
        "About": "ℹ️ About"
    }
    
    for page_name, page_icon in pages.items():
        if st.button(f"{page_icon} {page_name}"):
            st.session_state.current_page = page_name
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center;">
        <small>User ID: {st.session_state.user_id}</small>
    </div>
    """, unsafe_allow_html=True)

def show_dashboard():
    """Display the dashboard with user statistics and test history."""
    st.title("Dashboard")
    st.markdown("Welcome to the Parkinson's Disease Detection System. Track your symptoms and monitor your health over time.")
    
    # Get user history
    user_history = data_handler.get_user_history(st.session_state.user_id)
    
    # Calculate metrics
    total_tests = len(user_history) if not user_history.empty else 0
    last_test = user_history['timestamp'].max() if not user_history.empty else "N/A"
    
    # Calculate average risk score
    if not user_history.empty and 'risk_score' in user_history.columns:
        avg_risk = user_history['risk_score'].mean()
        risk_category = "High" if avg_risk > 0.6 else "Moderate" if avg_risk > 0.3 else "Low"
        risk_class = "high" if avg_risk > 0.6 else "moderate" if avg_risk > 0.3 else "low"
    else:
        avg_risk = 0
        risk_category = "N/A"
        risk_class = ""
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Last Test", last_test.split('T')[0] if last_test != "N/A" else "N/A")
    with col3:
        st.metric("Average Risk Score", f"{avg_risk*100:.1f}%" if avg_risk > 0 else "N/A")
    with col4:
        st.markdown("**Risk Category**")
        if risk_class:
            color = "#ff4b4b" if risk_class == "high" else "#ffa500" if risk_class == "moderate" else "#2ecc71"
            st.markdown(f"<div style='color: {color}; font-size: 1.5rem; font-weight: bold;'>{risk_category}</div>", 
                      unsafe_allow_html=True)
        else:
            st.markdown("N/A")
    
    # Show recent tests
    if not user_history.empty:
        st.subheader("Recent Tests")
        
        # Convert timestamp to datetime and format
        display_df = user_history.copy()
        if 'timestamp' in display_df.columns:
            display_df['Date'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            display_df = display_df.drop('timestamp', axis=1)
        
        # Reorder columns
        cols = ['Date', 'test_type', 'risk_score'] + [col for col in display_df.columns if col not in ['Date', 'test_type', 'risk_score']]
        display_df = display_df[cols]
        
        # Format risk score as percentage
        if 'risk_score' in display_df.columns:
            display_df['Risk Score'] = display_df['risk_score'].apply(lambda x: f"{x*100:.1f}%")
            display_df = display_df.drop('risk_score', axis=1)
        
        st.dataframe(display_df.head(5), use_container_width=True)
        
        # Show risk trend chart
        if len(user_history) > 1:
            st.subheader("Risk Trend Over Time")
            fig = visualizer.plot_risk_trend(user_history.to_dict('records'))
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No test history found. Complete a test to see your results here.")

class HandTracker(VideoTransformerBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarks_data = []
        self.test_start_time = None
        self.test_duration = 10  # seconds

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(img_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Store landmarks data for analysis
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                self.landmarks_data.append(landmarks)
                
                # Start test timer if not already started
                if self.test_start_time is None:
                    self.test_start_time = time.time()
                
                # Check if test duration is complete
                if time.time() - self.test_start_time >= self.test_duration:
                    self.analyze_tremor()
                    return img
        
        # Add timer display
        if self.test_start_time is not None:
            elapsed = time.time() - self.test_start_time
            remaining = max(0, self.test_duration - elapsed)
            cv2.putText(img, f"Time: {remaining:.1f}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return img
    
    def analyze_tremor(self):
        if len(self.landmarks_data) > 10:  # Ensure we have enough data
            try:
                landmarks_array = np.array(self.landmarks_data)
                tremor_analyzer = TremorAnalyzer()
                tremor_features = tremor_analyzer.calculate_tremor_features(landmarks_array)
                tremor_analysis = tremor_analyzer.analyze_tremor_severity(tremor_features)
                
                # Calculate test duration
                test_duration = time.time() - self.test_start_time
                
                # Calculate risk score
                test_results = {
                    'tremor': tremor_analysis,
                    'test_duration': test_duration,
                    'parameters': {
                        'sampling_rate': 30,
                        'window_size': 100,
                        'overlap': 0.5
                    },
                    'raw_data': landmarks_array
                }
                
                risk_calculator = RiskCalculator()
                risk_assessment = risk_calculator.calculate_risk(test_results)
                
                # Save to history
                data_handler.save_test_results(
                    st.session_state.user_id,
                    'tremor',
                    {
                        **tremor_analysis,
                        'test_duration': test_duration,
                        'risk_score': risk_assessment['overall_risk_score']
                    }
                )
                
                # Save results to session state
                st.session_state.landmarks_data = self.landmarks_data
                st.session_state.last_test_results = {
                    'type': 'tremor',
                    'results': tremor_analysis,
                    'test_duration': test_duration,
                    'timestamp': datetime.now().isoformat(),
                    'risk_assessment': risk_assessment
                }
                
                # Force UI update
                st.session_state.test_in_progress = False
                st.rerun()
                
            except Exception as e:
                st.error(f"Error analyzing tremor: {str(e)}")
                st.session_state.test_in_progress = False
                st.rerun()
            
            # Calculate risk score
            test_results = {
                'tremor': tremor_analysis,
                'test_duration': test_duration,
                'parameters': {
                    'sampling_rate': 30,
                    'window_size': 100,
                    'overlap': 0.5
                },
                'raw_data': landmarks_array
            }
            
            risk_calculator = RiskCalculator()
            risk_assessment = risk_calculator.calculate_risk(test_results)
            
            # Save to history
            data_handler.save_test_results(
                st.session_state.user_id,
                'tremor',
                {
                    **tremor_analysis,
                    'test_duration': test_duration,
                    'risk_score': risk_assessment['overall_risk_score']
                }
            )
            
            st.session_state.last_test_results['risk_assessment'] = risk_assessment
            st.session_state.test_in_progress = False
            st.rerun()

def show_tremor_test():
    """Display the tremor test interface with real-time camera feed."""
    st.title("Tremor Test")
    st.markdown("""
    This test analyzes hand tremors, a common symptom of Parkinson's disease. 
    Follow the instructions below to complete the test.
    """)
    
    # Test instructions
    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        1. Position your hand in front of your device's camera.
        2. Keep your hand steady with fingers slightly spread.
        3. Click 'Start Test' and hold the position for 10 seconds.
        4. Try to keep your hand as still as possible.
        5. The system will analyze any tremors in your hand movements.
        """)
    
    # Initialize session state for test status
    if 'test_in_progress' not in st.session_state:
        st.session_state.test_in_progress = False
    
    # Start/Stop test buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎥 Start Test", disabled=st.session_state.test_in_progress):
            st.session_state.test_in_progress = True
            st.session_state.landmarks_data = []
            st.session_state.test_start_time = None
            st.rerun()
    
    # Camera feed and test display
    if st.session_state.test_in_progress:
        # Initialize test start time if not set
        if 'test_start_time' not in st.session_state:
            st.session_state.test_start_time = time.time()
        
        # Calculate test duration
        test_duration = time.time() - st.session_state.test_start_time
        
        # Show progress
        progress = min(test_duration / 10, 1.0)
        st.progress(progress)
        st.info(f"Test in progress: {min(int(test_duration), 10)}/10 seconds")
        
        # Display the camera feed with hand tracking
        ctx = webrtc_streamer(
            key="tremor-test",
            video_processor_factory=HandTracker,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Check if test duration is complete
        if test_duration >= 10 and 'last_test_results' not in st.session_state:
            st.session_state.test_in_progress = False
            st.rerun()
        
        # Add a stop button
        if st.button("🛑 Stop Test"):
            st.session_state.test_in_progress = False
            st.rerun()
    
    # Display results if available
    if 'last_test_results' in st.session_state and st.session_state.last_test_results['type'] == 'tremor':
        results = st.session_state.last_test_results['results']
        risk = st.session_state.last_test_results.get('risk_assessment', {'risk_category': 'N/A'})
        
        st.subheader("Test Results")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tremor Score", f"{results.get('tremor_score', 0)*100:.1f}%")
        with col2:
            st.metric("Severity", results.get('severity', 'N/A'))
        with col3:
            st.metric("Risk Level", risk.get('risk_category', 'N/A'))
        
        # Show tremor analysis visualization if available
        if visualizer:
            st.plotly_chart(visualizer.plot_tremor_analysis(results), use_container_width=True)
        
        # Show risk assessment
        st.subheader("Risk Assessment")
        st.markdown(f"**{risk['risk_category']} Risk**")
        st.progress(risk['overall_risk_score'])
        st.markdown(risk['explanation'])
        st.markdown(f"**Recommendation:** {risk['recommendation']}")

def show_tap_test():
    """Display the finger tap test interface."""
    st.title("Finger Tap Test")
    st.markdown("""
    This test measures bradykinesia (slowness of movement) by analyzing your finger tapping.
    Follow the instructions below to complete the test.
    """)
    
    # Test instructions
    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        1. Place your hand on a flat surface in front of your device's camera.
        2. Tap your index finger and thumb together repeatedly.
        3. Click 'Start Test' and tap for 10 seconds.
        4. Try to maintain a consistent rhythm and amplitude.
        5. The system will analyze your tapping pattern.
        """)
    
    # Test controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👆 Start Tap Test"):
            st.session_state.test_in_progress = True
            st.session_state.tap_test_start = time.time()
            st.session_state.tap_count = 0
            
    with col2:
        if st.button("⏹️ Stop Test", disabled=not st.session_state.get('test_in_progress', False)):
            st.session_state.test_in_progress = False
            
            # Simulate test results (in a real app, this would come from the camera feed analysis)
            test_duration = time.time() - st.session_state.tap_test_start
            
            # Generate sample tap data
            tap_analyzer = TapAnalyzer()
            sample_data = np.random.randn(300, 3)  # Simulated finger tip positions
            tap_results = tap_analyzer.detect_taps(sample_data)
            tap_analysis = tap_analyzer.analyze_tap_performance(tap_results)
            
            # Calculate risk score
            risk_calculator = RiskCalculator()
            risk_assessment = risk_calculator.calculate_risk(
                {'tap': tap_analysis},
                {'age': 50}  # Example age
            )
            
            # Save to history
            data_handler.save_test_results(
                st.session_state.user_id,
                'tap',
                {
                    **tap_analysis,
                    'test_duration': test_duration,
                    'risk_score': risk_assessment['overall_risk_score']
                }
            )
            
            st.session_state.last_test_results = {
                'type': 'tap',
                'results': tap_analysis,
                'risk_assessment': risk_assessment,
                'timestamp': datetime.now().isoformat()
            }
            
            st.experimental_rerun()
    
    # Display test status
    if st.session_state.get('test_in_progress', False) and 'tap_test_start' in st.session_state:
        test_duration = time.time() - st.session_state.tap_test_start
        progress = min(test_duration / 10, 1.0)  # 10-second test
        
        st.progress(progress)
        st.info(f"Test in progress: {min(int(test_duration), 10)}/10 seconds")
        
        # Simulate tap counting (in a real app, this would come from the camera)
        if test_duration < 10:
            st.session_state.tap_count = int(test_duration * 3 + np.random.normal(0, 0.5))
            st.metric("Tap Count", st.session_state.tap_count)
        else:
            st.session_state.test_in_progress = False
            st.rerun()
    
    # Display results if available
    if 'last_test_results' in st.session_state and st.session_state.last_test_results['type'] == 'tap':
        results = st.session_state.last_test_results['results']
        risk = st.session_state.last_test_results['risk_assessment']
        
        st.subheader("Test Results")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Tap Count", results.get('tap_count', 0))
        with col2:
            st.metric("Bradykinesia Score", f"{results.get('bradykinesia_score', 0)*100:.1f}%")
        with col3:
            st.metric("Severity", results.get('severity', 'N/A'))
        
        # Show tap analysis visualization
        st.plotly_chart(visualizer.plot_tap_analysis(results), use_container_width=True)
        
        # Show risk assessment
        st.subheader("Risk Assessment")
        st.markdown(f"**{risk['risk_category']} Risk**")
        st.progress(risk['overall_risk_score'])
        st.markdown(risk['explanation'])
        st.markdown(f"**Recommendation:** {risk['recommendation']}")

def show_symptom_survey():
    """Display the symptom survey form."""
    st.title("Symptom Survey")
    st.markdown("""
    Please complete this survey about your symptoms. This information will help assess your risk 
    of Parkinson's disease and track changes over time.
    """)
    
    with st.form("symptom_survey"):
        st.subheader("Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 50)
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
            
        with col2:
            height = st.number_input("Height (cm)", 100, 250, 170)
            weight = st.number_input("Weight (kg)", 30, 200, 70)
        
        st.subheader("Symptoms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tremor = st.checkbox("Tremor (shaking) at rest")
            rigidity = st.checkbox("Muscle stiffness or rigidity")
            bradykinesia = st.checkbox("Slowness of movement")
            
        with col2:
            postural = st.checkbox("Balance problems or falls")
            gait = st.checkbox("Shuffling walk or freezing of gait")
            micrographia = st.checkbox("Small, cramped handwriting (micrographia)")
        
        st.subheader("Additional Information")
        
        family_history = st.radio(
            "Do you have a family history of Parkinson's disease?",
            ["No", "Yes, first-degree relative", "Yes, other relative", "Not sure"]
        )
        
        symptom_duration = st.select_slider(
            "How long have you been experiencing these symptoms?",
            options=["Less than 3 months", "3-6 months", "6-12 months", "1-2 years", "More than 2 years"]
        )
        
        impact = st.select_slider(
            "How much do these symptoms impact your daily life?",
            options=["Not at all", "Slightly", "Moderately", "Severely", "Extremely"]
        )
        
        notes = st.text_area("Additional notes about your symptoms:")
        
        submitted = st.form_submit_button("Submit Survey")
        
        if submitted:
            # Prepare survey data
            survey_data = {
                'age': age,
                'gender': gender,
                'height_cm': height,
                'weight_kg': weight,
                'symptoms': {
                    'tremor': tremor,
                    'rigidity': rigidity,
                    'bradykinesia': bradykinesia,
                    'postural_instability': postural,
                    'gait_difficulties': gait,
                    'micrographia': micrographia
                },
                'family_history': family_history,
                'symptom_duration': symptom_duration,
                'impact': impact,
                'notes': notes,
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate risk score based on symptoms
            risk_calculator = RiskCalculator()
            risk_assessment = risk_calculator.calculate_risk(
                {'symptoms': survey_data['symptoms']},
                {
                    'age': age,
                    'gender': gender.lower(),
                    'family_history': 'yes' if 'yes' in family_history.lower() else 'no'
                }
            )
            
            # Save survey results
            data_handler.save_test_results(
                st.session_state.user_id,
                'survey',
                {
                    **survey_data,
                    'risk_score': risk_assessment['overall_risk_score']
                }
            )
            
            st.session_state.last_survey = {
                'data': survey_data,
                'risk_assessment': risk_assessment,
                'timestamp': datetime.now().isoformat()
            }
            
            st.success("Thank you for completing the survey! Your responses have been recorded.")
            
            # Show risk assessment
            st.subheader("Preliminary Risk Assessment")
            st.markdown(f"**{risk_assessment['risk_category']} Risk**")
            st.progress(risk_assessment['overall_risk_score'])
            st.markdown(risk_assessment['explanation'])
            st.markdown(f"**Recommendation:** {risk_assessment['recommendation']}")

def show_results_history():
    """Display test history and detailed results."""
    st.title("Results & History")
    
    # Get user history
    user_history = data_handler.get_user_history(st.session_state.user_id)
    
    if user_history.empty:
        st.info("No test history found. Complete a test to see your results here.")
        return
    
    # Show summary statistics
    st.subheader("Test History Summary")
    
    # Calculate metrics
    total_tests = len(user_history)
    test_types = user_history['test_type'].value_counts().to_dict()
    
    if 'risk_score' in user_history.columns:
        avg_risk = user_history['risk_score'].mean()
        risk_trend = "increasing" if len(user_history) > 1 and user_history['risk_score'].iloc[-1] > user_history['risk_score'].iloc[0] else "stable"
    else:
        avg_risk = 0
        risk_trend = "N/A"
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Test Types", ", ".join([f"{k} ({v})" for k, v in test_types.items()]))
    with col3:
        st.metric("Average Risk", f"{avg_risk*100:.1f}%" if avg_risk > 0 else "N/A")
    with col4:
        st.metric("Risk Trend", risk_trend.capitalize())
    
    # Show detailed test history
    st.subheader("Detailed Test History")
    
    # Format the data for display
    display_df = user_history.copy()
    
    # Convert timestamp to datetime and format
    if 'timestamp' in display_df.columns:
        display_df['Date'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df.drop('timestamp', axis=1)
    
    # Reorder columns
    cols = ['Date', 'test_type']
    if 'risk_score' in display_df.columns:
        cols.append('risk_score')
    
    # Add remaining columns
    cols += [col for col in display_df.columns if col not in cols]
    display_df = display_df[cols]
    
    # Format risk score as percentage
    if 'risk_score' in display_df.columns:
        display_df['Risk Score'] = display_df['risk_score'].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
        display_df = display_df.drop('risk_score', axis=1)
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)
    
    # Show risk trend chart
    if len(user_history) > 1 and 'risk_score' in user_history.columns:
        st.subheader("Risk Trend Over Time")
        fig = visualizer.plot_risk_trend(user_history.to_dict('records'))
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Export data option
    st.subheader("Export Data")
    st.markdown("Download your test history for your records or to share with your healthcare provider.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export as CSV"):
            export_path = data_handler.export_user_data(st.session_state.user_id, 'csv')
            if export_path:
                with open(export_path, 'rb') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f,
                        file_name=f"parkinsons_risk_assessment_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime='text/csv'
                    )
    
    with col2:
        if st.button("📊 Export as JSON"):
            export_path = data_handler.export_user_data(st.session_state.user_id, 'json')
            if export_path:
                with open(export_path, 'rb') as f:
                    st.download_button(
                        label="Download JSON",
                        data=f,
                        file_name=f"parkinsons_risk_assessment_{datetime.now().strftime('%Y%m%d')}.json",
                        mime='application/json'
                    )

def show_about():
    """Display information about the application."""
    st.title("About Parkinson's Disease Detection System")
    
    st.markdown("""
    ## Overview
    This application is designed to help with the early detection and monitoring of Parkinson's disease 
    through a series of simple tests and surveys. It is not a diagnostic tool but can help identify 
    potential symptoms that may warrant further medical evaluation.
    
    ## Features
    - **Tremor Test**: Analyzes hand tremors using your device's camera
    - **Tap Test**: Measures bradykinesia (slowness of movement) through finger tapping
    - **Symptom Survey**: Captures self-reported symptoms and medical history
    - **Results & History**: Tracks your test results over time
    
    ## How It Works
    1. Complete the tests and surveys as directed
    2. The system analyzes your movements and responses
    3. Receive a risk assessment based on your results
    4. Track changes in your symptoms and risk over time
    
    ## Privacy
    - All data is stored locally on your device
    - No personal health information is shared without your consent
    - You can export your data at any time
    
    ## Disclaimer
    This application is for informational purposes only and is not a substitute for professional medical 
    advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health 
    provider with any questions you may have regarding a medical condition.
    
    ## Version
    Version 1.0.0
    
    ## Contact
    For questions or feedback, please contact support@example.com
    """)

# Main app routing
if st.session_state.current_page == "Dashboard":
    show_dashboard()
elif st.session_state.current_page == "Tremor Test":
    show_tremor_test()
elif st.session_state.current_page == "Tap Test":
    show_tap_test()
elif st.session_state.current_page == "Symptom Survey":
    show_symptom_survey()
elif st.session_state.current_page == "Results & History":
    show_results_history()
elif st.session_state.current_page == "About":
    show_about()
