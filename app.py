import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import requests

def classify_emotion(text, token):
    """Classify the emotion of the given text using the Hugging Face Inference API."""
    api_url = "https://api-inference.huggingface.co/models/michellejieli/emotion_text_classifier"
    
    # Prepare the headers with the API token
    headers = {"Authorization": f"Bearer {token}"}

    # Prepare the payload
    payload = {"inputs": text, "options": {"wait_for_model": True}, "parameters": {"return_all_scores": True}}

    # Make the request
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to classify emotion: {response.text}")

def plot_emotion_radar(scores, labels):
    """Plot the emotion radar chart based on classification scores."""
    scores = np.array(scores)
    scores = np.concatenate((scores, [scores[0]]))  # Close the radar chart
    labels = np.array(labels)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the radar chart

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='magenta', alpha=0.6)
    ax.plot(angles, scores, color='magenta', marker='o')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='navy', size=8)
    ax.set_yticks([])
    plt.title('Emotion Radar', size=14, color='navy', y=1.1)
    return fig

def main():
    """Run the main Streamlit app."""
    st.title('Journal Emotion Radar')

    # Attempt to access the Hugging Face API key from secrets.toml or sidebar
    token = st.secrets["HuggingFace"]["HUGGINGFACE_TOKEN"] if "huggingface" in st.secrets else None
    if not token:
        with st.sidebar:
            token = st.text_input("Enter your Hugging Face API token:", key="api_key", type="password")
    
    # Main area for user text input
    user_input = st.text_area("Enter your journal entry or text here:", height=150)

    # Button to classify text
    submit = st.button('Submit')

    if submit:
        if not token:
            st.warning('Please enter your Hugging Face API token in the sidebar.', icon="⚠️")
        elif not user_input:
            st.warning('Please enter some text to analyze.', icon="⚠️")
        else:
            # Classify the emotion
            try:
                results = classify_emotion(user_input, token)
                
                # Extract labels and scores for plotting
                labels = [result['label'] for result in results[0]]
                scores = [result['score'] for result in results[0]]

                # Plot and display the radar chart
                fig = plot_emotion_radar(scores, labels)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error in emotion classification: {e}")

if __name__ == "__main__":
    main()
