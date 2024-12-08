import os
import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr
import google.generativeai as genai
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Configure Google Generative AI
GOOGLE_API_KEY = "AIzaSyBxv2Ssm0SZCEGx7oJJwW5plWXZKnTUQvQ"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Load the trained model
model = tf.keras.models.load_model('final_model.h5')  # Update this path
class_names = ['fs', 'nt', 'se', 'tf']  # Update class names if needed
class_to_number = {
    'fs': "B (5-7 years)",
    'nt': "D (10-12 years)", 
    'se': "C (7-9 years)",
    'tf': "A (3-5 years)"
}

# Create necessary directories
SPECTROGRAM_FOLDER = 'spectrograms'
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

# Initialize prediction history DataFrame
prediction_history = pd.DataFrame(columns=[
    'Name', 
    'Real Age', 
    'Real Age Group', 
    'Predicted Age Group', 
    'Prediction Confidence'
])

# Initial Prompt for Chatbot
INITIAL_PROMPT = "Hello! I'm here to assist you with understanding your child's speech development. Please provide details about your child's voice recording or ask any questions about speech milestones."

def audio_to_spectrogram(audio_path, save_path):
    """Convert audio file to spectrogram image."""
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Generate Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Plot and save the spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error in audio_to_spectrogram: {e}")
        raise

def get_age_group(age):
    """Determine age group based on child's actual age."""
    if 3 <= age < 5:
        return 'A (3-5 years)'
    elif 5 <= age < 7:
        return 'B (5-7 years)'
    elif 7 <= age < 9:
        return 'C (7-9 years)'
    elif 10 <= age < 12:
        return 'D (10-12 years)'
    else:
        return 'Unknown'

def predict_age(audio_file, child_name, child_age):
    """Predict age based on audio input and store prediction history."""
    global prediction_history
    
    if audio_file is None:
        return "No audio file provided", "0%", prediction_history
    
    try:
        # Create a unique name for the spectrogram image
        spectrogram_image_name = f"spectrogram_{np.random.randint(1000, 9999)}.png"
        spectrogram_path = os.path.join(SPECTROGRAM_FOLDER, spectrogram_image_name)
        
        # Convert audio to spectrogram and save as an image
        audio_to_spectrogram(audio_file, spectrogram_path)
        
        # Load the spectrogram image for prediction
        img = tf.keras.utils.load_img(spectrogram_path, target_size=(200, 200))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        
        # Make prediction using the loaded model
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = float(100 * np.max(score))  # Convert to a Python float
        
        # Clean up the spectrogram image
        os.remove(spectrogram_path)
        
        # Get the predicted age category with group
        result = class_to_number[predicted_class]
        
        # Determine the real age group
        real_age_group = get_age_group(child_age)
        
        # Create a new prediction record
        new_prediction = pd.DataFrame({
            'Name': [child_name],
            'Real Age': [child_age],
            'Real Age Group': [real_age_group],
            'Predicted Age Group': [result],
            'Prediction Confidence': [f"{confidence:.2f}%"]
        })
        
        # Append to prediction history
        prediction_history = pd.concat([prediction_history, new_prediction], ignore_index=True)
        
        return f"Predicted Age Category: {result}", f"Confidence: {confidence:.2f}%", prediction_history
    
    except Exception as e:
        print(f"Error in predict_age: {e}")
        return f"Prediction Error: {str(e)}", "0%", prediction_history

def chatbot_response(message, history):
    """Generate context-aware chatbot responses using Google Generative AI"""
    # Initialize the model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Prepare conversation history for context
    context = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in history
    ])
    
    # Construct the full prompt
    full_prompt = f"""
        Role: You are Dr. Speech, a virtual speech specialist assisting parents in understanding their child's speech development based on voice analysis results.

        Project Context: This tool uses an AI model that predicts a child's age group based on their voice recording. If the model predicts an age group much higher than the child's actual age, it suggests leading speech development. If it predicts a lower age group, it suggests lagging speech development. The group and their respective age are given below.
        Group A: 3-5 years
        Group B: 5-7 years
        Group C: 7-9 years
        Group D: 10-12 years

        Conversation History:
        {context}

        User's Current Message:
        {message}

        Guidelines for Response:
        - Provide professional, context-aware responses as a speech specialist.
        - Offer specific insights on child speech development.
        - Explain the implications of the model's age prediction in terms of speech growth (leading or lagging).
        - Provide suggestions, tips, or advice related to speech development.
        - Be supportive, empathetic, and informative.
        - Keep responses concise and clear.
        """
    
    try:
        # Generate response
        response = model.generate_content(full_prompt)
        # Append the new message to the history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response.text})
        return history, ""  # Return updated history and clear the input
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again.", ""

# Create Gradio interface with tabs
# (Previous imports and code remain the same)

# Create Gradio interface with tabs
with gr.Blocks(css="""
    /* Target the tab container */
    .gradio-container .tab-container {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    
    /* Center the tab buttons */
    .gradio-container .tab-container button {
        margin: 0 10px;
    }
""") as demo:
    gr.Markdown("""
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 20px;">
        <h1>SONIC PINNACLE<br>Advancing Age-Linked Speech Analysis through AI</h1>
    </div>
    """, elem_id="title")
    
    gr.Markdown("""
    <div align="center" style="max-width: 80%; margin: 0 auto;">
    Sonic Pinnacle is an innovative AI-powered tool designed to assess and analyze children's speech development. 
    By leveraging advanced machine learning algorithms, we provide insights into speech patterns, 
    helping parents and professionals understand a child's linguistic progression.
    Our cutting-edge technology transforms voice recordings into detailed spectral analyses, 
    offering precise age-group predictions and developmental insights.
    </div>
    """, elem_id="description")
    
    with gr.Tabs():
        with gr.TabItem("Age Prediction", id="prediction"):
            with gr.Row():
                with gr.Column():
                    # Rearranged input fields: audio first, then name, then age
                    audio_input = gr.Audio(type="filepath", label="Upload Child's Voice Recording")
                    child_name_input = gr.Textbox(label="Child's Name")
                    child_age_input = gr.Number(label="Child's Age")
                    predict_btn = gr.Button("Predict Age Category")
                
                with gr.Column():
                    age_output = gr.Textbox(label="Predicted Age Category")
                    confidence_output = gr.Textbox(label="Prediction Confidence")
            
            # Prediction history dataframe
            prediction_history_output = gr.Dataframe(
                headers=['Name', 'Real Age', 'Real Age Group', 'Predicted Age Group', 'Prediction Confidence'],
                label="Prediction History"
            )
            
            predict_btn.click(
                fn=predict_age, 
                inputs=[audio_input, child_name_input, child_age_input], 
                outputs=[age_output, confidence_output, prediction_history_output]
            )
        
        with gr.TabItem("Speech Development Support", id="chatbot"):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": INITIAL_PROMPT}],  # Start with initial prompt
                label="Speech Development Assistant",
                type="messages"  # Updated to resolve deprecation warning
            )
            msg = gr.Textbox(label="Your Question")
            submit = gr.Button("Send")
            clear = gr.Button("Clear")
            
            submit.click(
                fn=chatbot_response, 
                inputs=[msg, chatbot], 
                outputs=[chatbot, msg]
            )
            
            clear.click(
                lambda: [{"role": "assistant", "content": INITIAL_PROMPT}],  # Reset to initial prompt
                None, 
                [chatbot], 
                queue=False
            )

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()