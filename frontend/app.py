import streamlit as st
import os
import requests

# Constants
FASTAPI_URL = "http://127.0.0.1:8000/"  # FastAPI base URL
LOGO_PATH = "frontend/assets/app-logo.png"  # Ensure this matches your actual filename

def configure_page():
    """Sets up the Streamlit page configuration."""
    st.set_page_config(layout="wide", page_title="Nvidia Assistant")

def get_page_title():
    """Fetch page title from FastAPI."""
    try:
        response = requests.get(FASTAPI_URL)
        if response.status_code == 200:
            return response.json().get("message", "Nvidia Assistant")
        else:
            return "Nvidia Research Assistant (Fallback)"
    except requests.exceptions.RequestException:
        return "Nvidia Research Assistant (Offline)"

def display_sidebar():
    """Displays the sidebar with logo and welcome message."""
    with st.sidebar:

        # Display Logo
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.warning(f"⚠️ Logo not found! Expected path: {LOGO_PATH}")

        # Welcome Text
        st.markdown("## Welcome to **Nvidia Research Assistant!**")

        # Options for User Selection
        st.markdown("### Please select an option:")
        options = [
            "Summarize",
            "Data-driven visuals",
            "Real-time insights",
            "Complete report"
        ]
        
        action = st.radio("Select an option", options)

        # Time Duration Selection (Year)
        st.markdown("### Select the Time Duration:")
        
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

        # Collect the selections into a tuple
        user_selection = (action, years, quarters_dict)

        return user_selection

import streamlit as st

def display_main_content(user_selection):
    """Displays the main content on the page."""
    page_title = get_page_title()  # Replace with your actual page title logic
    st.title(page_title)

    # Unpack user_selection into separate variables for readability
    action, years, quarters_dict = user_selection

    # Create a container to hold the elements
    with st.container():
        # Create two columns for left-hand side (LHS) and right-hand side (RHS)
        col1, col2 = st.columns(2)

        # Left column (LHS) for Action
        with col1:
            st.markdown(f"##### Action: {action}")

        # Right column (RHS) for Year and Quarter
        with col2:
            st.write("##### Year and Quarter:")
            for year in years:
                quarters = ', '.join(quarters_dict.get(year, []))
                st.write(f"###### Year: {year} | Quarter: {quarters}")

    # Create a text box (text area) for user input
    prompt = st.text_area("#####", height=150, placeholder="Example: Analyze sales trend for given duaration")  # height sets initial size

    # Button to submit the input
    submit_button = st.button("Submit")

    # When the submit button is pressed
    if submit_button:
        # Check if a year and at least one quarter is selected
        if not years:
            st.write("##### ⚠️ Please select at least one year and one quarter before submitting.")
        elif all(len(quarters_dict.get(year, [])) == 0 for year in years):
            st.write("##### ⚠️ Please select at least one quarter before submitting.")
        elif not prompt:
            st.write("##### ⚠️ Please enter a prompt before submitting.")
        else:
            st.write(f"###### Prompt entered: {prompt}")
            st.write("###### Year and Quarter Information:")
            # Display year and quarter information after submission
            for year in years:
                st.write(f"{year}: {quarters_dict.get(year, [])}")
            print("Years ",years)
            print(action)
            if action == "Summarize":
                # Prepare request payload
                request_data = {
                    "query": prompt,
                    "year_quarter_dict": {str(year): quarters_dict[year] for year in years}
                }
                print(request_data)

                # Send request to FastAPI
                try:
                    response = requests.post(f"{FASTAPI_URL}summarize_using_pinecone", json=request_data)
                    if response.status_code == 200:
                        result = response.json().get("response", "No response received.")
                        st.success("✅ Response received:")
                        st.write(result)
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
