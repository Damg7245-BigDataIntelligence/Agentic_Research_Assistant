import streamlit as st
import requests
from datetime import datetime
import re

# Constants
FASTAPI_URL = "http://34.85.173.233:8000/"  # FastAPI base URL

def configure_page():
    """Sets up the Streamlit page configuration."""
    st.set_page_config(
        layout="wide", 
        page_title="NVIDIA Research Assistant",
        page_icon="🔍",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS for better styling
    st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #76b900;
            background-image: linear-gradient(315deg, #76b900 0%, #1a1a1a 74%);
            color: white;
        }
        .stRadio > div {
            padding: 10px;
            border-radius: 5px;
            background-color: rgba(255, 255, 255, 0.1);
        }
        .stRadio > div:hover {
            background-color: rgba(255, 255, 255, 0.2);
        }
        .stButton>button {
            background-color: #76b900;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #8fd400;
        }
        h1, h2, h3 {
            color: #76b900;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
        .stTextArea textarea {
            border-radius: 5px;
            border: 1px solid #76b900;
        }
        .nvidia-card {
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .nvidia-header {
            color: #76b900;
            font-weight: bold;
        }
        .expanded-viz {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #ddd;
        }
        
        .download-button {
            text-align: center;
            margin-top: 10px;
        }
        
        .download-button a {
            background-color: #76b900;
            color: white;
            padding: 8px 16px;
            border-radius: 5px;
            text-decoration: none;
            display: inline-block;
        }
        
        .download-button a:hover {
            background-color: #8fd400;
        }
    </style>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Displays the sidebar with logo and welcome message."""
    with st.sidebar:
        # NVIDIA Logo
        st.image("https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-vert-500x200-2c50-d.png", width=200)
        
        # Welcome text with divider
        st.markdown("## Welcome to NVIDIA Research Assistant")
        st.markdown("---")
        
        # Research mode selection with better styling
        st.markdown("### 🔬 **Select Research Mode**")
        mode_descriptions = {
            "Pinecone RAG Search": "Search through NVIDIA quarterly reports and financial data",
            "Web Search": "Explore recent news and web content about NVIDIA",
            "Snowflake RAG Search": "Search through NVIDIA quarterly reports and financial data",
            "Combined Research": "Comprehensive analysis using all available data sources"
        }
        
        # Present options with descriptions
        action = st.radio(
            "Choose your research approach:",
            options=list(mode_descriptions.keys()),
            key="research_mode"
        )
        
        # Show selected mode description
        st.markdown(f"<div style='background-color:rgba(118, 185, 0, 0.1); padding:10px; border-radius:5px; margin-bottom:15px;'><small>{mode_descriptions[action]}</small></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        # Dictionary to store quarters for each year
        quarters_dict = {}
        
        # Quarter selection with better UI
        # Time Duration Selection (Year)
        st.markdown("### 📊 **Select the Time Duration:**")
        
        # Multi-select Year selection
        years = st.multiselect("Select Year(s)", [2020, 2021, 2022, 2023, 2024, 2025], default=[2024])
        
        # Dictionary to store quarters for each year
        quarters_dict = {}

        # For each selected year, dynamically show checkboxes for quarters
        for year in years[::-1]:
            st.markdown(f"### Select Quarters for {year}:")
            quarters = []
            if st.checkbox("Q1", key=f"{year}_Q1"):
                quarters.append("1")
            if st.checkbox("Q2", key=f"{year}_Q2"):
                quarters.append("2")
            if st.checkbox("Q3", key=f"{year}_Q3"):
                quarters.append("3")
            if st.checkbox("Q4", key=f"{year}_Q4"):
                quarters.append("4")
            
            # Save selected quarters for the specific year
            quarters_dict[year] = quarters

        # Show help information
        with st.expander("Need Help?", expanded=False):
            st.markdown("""
            **How to use this tool:**
            1. Select your research mode based on what you need
            2. Choose the years and quarters of interest
            3. Type your query in the main panel
            4. Click Submit to get your research insights
            
            For detailed financial analysis, select specific quarters.
            For trend analysis, select multiple years.
            """)
        
        # Collect selections
        user_selection = (action, years, quarters_dict)
        return user_selection

def display_visualization(segment, viz_index):
    """Helper function to display visualization with proper formatting"""
    url_match = re.search(r'!\[(.*?)\]\((.*?)\)', segment)
    if url_match:
        title = url_match.group(1)
        image_url = url_match.group(2)
        
        # Create columns for visualization layout
        col1, col2 = st.columns([4, 1])
        
        with col1:
            try:
                # Display the image
                st.image(image_url, caption=title, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load visualization: {str(e)}")
                st.markdown(f"[View Image Directly]({image_url})")
        
        with col2:
            # Create a unique key for the expander state
            expand_state = f"expand_state_{viz_index}"
            if expand_state not in st.session_state:
                st.session_state[expand_state] = False
            
            
        # Show expanded view if state is True
        if st.session_state.get(expand_state, False):
            st.markdown("<div class='expanded-viz'>", unsafe_allow_html=True)
            st.image(image_url, caption=f"{title} (Expanded View)", use_column_width=True)
            st.markdown("<div class='download-button'>", unsafe_allow_html=True)
            st.markdown(f"<a href='{image_url}' target='_blank'>📥 Download Image</a>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

def display_main_content(user_selection):
    """Displays the main content area with enhanced UI."""
    # Create a branded header
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0; flex-grow: 1;">NVIDIA Research Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Horizontal line
    st.markdown("<hr style='margin-bottom: 30px;'>", unsafe_allow_html=True)

    # Unpack user_selection
    action, years, quarters_dict = user_selection

    # Create a card-like container for the selection summary
    
    # Create two columns for the setup summary
    col1, col2 = st.columns(2)
    
    # Left column for Research Mode
    with col1:
        icon_map = {
            "Pinecone RAG Search": "📚",
            "Web Search": "🌐",
            "Snowflake Search": "🔄",
            "Combined Research": "🔄"
        }
        icon = icon_map.get(action, "🔍")
        st.markdown(f"##### {icon} Research Mode")
        st.info(action)

    # Right column for Time Period
    with col2:
        st.markdown("##### 📅 Time Period")
        if years:
            for year in sorted(years):
                quarters = [f"Q{q}" for q in quarters_dict.get(year, [])]
                if quarters:
                    quarters_str = ", ".join(quarters)
                    st.success(f"{year}: {quarters_str}")
                else:
                    st.warning(f"{year}: No quarters selected")
        else:
            st.warning("No time period selected")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a card for the query input
    st.markdown("""
    <div class="nvidia-card">
        <h3 class="nvidia-header">Research Query</h3>
    """, unsafe_allow_html=True)
    
    # Research query input with better placeholder
    placeholder_examples = {
        "Pinecone RAG Search": "Example: How did NVIDIA's revenue growth compare across different business segments in the selected quarters?",
        "Web Search": "Example: What are the latest developments in NVIDIA's AI technology and market position?",
        "Combined Research": "Example: Analyze NVIDIA's financial performance and recent strategic initiatives in the AI market"
    }
    
    placeholder = placeholder_examples.get(action, "Enter your research question here...")
    
    prompt = st.text_area(
        "What would you like to research about NVIDIA?",
        height=120,
        placeholder=placeholder,
        key="research_query"
    )
    
    # Sample queries for user guidance
    with st.expander("Sample Queries", expanded=False):
        st.markdown("""
        **Financial Analysis Examples:**
        - How has NVIDIA's revenue changed over the selected quarters?
        - Compare NVIDIA's gross margin trends across the past 2 years
        - What were the key growth drivers mentioned in the 2024 Q1 report?
        
        **Market Research Examples:**
        - What is NVIDIA's position in the AI chip market?
        - How are NVIDIA's data center products performing compared to competitors?
        - What are analysts saying about NVIDIA's future growth prospects?
        """)
    
    # Submit button with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button("🔎 Submit Research Query", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process the submission
    if submit_button:
        # Input validation
        if not prompt:
            st.error("⚠️ Please enter a research query before submitting.")
        elif action in ["Pinecone RAG Search", "Combined Research"] and not years:
            st.error("⚠️ Please select at least one year for this research mode.")
        elif action in ["Pinecone RAG Search", "Combined Research"] and all(len(quarters_dict.get(year, [])) == 0 for year in years):
            st.error("⚠️ Please select at least one quarter for this research mode.")
        else:
            # Create year_quarter_dict for API request
            year_quarter_dict = {str(year): quarters_dict[year] for year in years}
            
            # Map the action to the correct mode for our backend
            mode_mapping = {
                "Pinecone RAG Search": "pinecone",
                "Web Search": "web_search",
                "Snowflake RAG Search": "snowflake",
                "Combined Research": "combined"
            }
            
            # Research request
            research_request = {
                "query": prompt,
                "year_quarter_dict": year_quarter_dict,
                "mode": mode_mapping[action]
            }
            
            # Display a professional loading animation
            with st.spinner(f"Researching your query using {action} mode..."):
                try:
                    response = requests.post(f"{FASTAPI_URL}research", json=research_request)
                    
                    if response.status_code == 200:
                        data = response.json()
                        result = data.get("result", "No result received.")
                        processing_time = data.get("processing_time", 0)
                        
                        # Create a results card
                        st.markdown("""
                        <div class="nvidia-card" style="background-color: #f9f9f9;">
                            <h3 class="nvidia-header">Research Results</h3>
                        """, unsafe_allow_html=True)
                        
                        # Display processing metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"✅ Results generated in {processing_time:.2f} seconds")
                        with col2:
                            current_time = datetime.now().strftime("%I:%M %p, %b %d, %Y")
                            st.info(f"🕒 Generated at: {current_time}")
                        
                        # Display result in a well-formatted container
                        st.markdown("""
                        <div style="background-color: white; padding: 20px; border-radius: 5px; border-left: 5px solid #76b900; margin-top: 20px;">
                        """, unsafe_allow_html=True)
                        
                        # Enhanced visualization handling
                        if "![" in result and "](" in result:
                            # Split content into text and image segments
                            segments = re.split(r'(!\[.*?\]\(.*?\))', result)
                            
                            viz_counter = 0  # Initialize counter for unique keys
                            for segment in segments:
                                if segment.startswith('!['):
                                    display_visualization(segment, viz_counter)
                                    viz_counter += 1  # Increment counter for next visualization
                                else:
                                    st.markdown(segment)
                        else:
                            # If no images found, display the text as is
                            st.markdown(result)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to connect to API: {e}")
        
def main():
    """Main function to run the Streamlit app."""
    configure_page()
    
    user_selection = display_sidebar()
    display_main_content(user_selection)
    
if __name__ == "__main__":
    main()
