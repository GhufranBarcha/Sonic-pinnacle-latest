import os
import numpy as np
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
    'fs': "B",
    'nt': "D",
    'se': "C",
    'tf': "A"
}

# Create necessary directories
SPECTROGRAM_FOLDER = 'spectrograms'
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

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

def predict_age(audio_file):
    """Predict age based on audio input."""
    if audio_file is None:
        return "No audio file provided", "0%"
    
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
        
        # Get the predicted age category
        result = class_to_number[predicted_class]
        
        return f"Predicted Age Category: {result}", f"Confidence: {confidence:.2f}%"
    
    except Exception as e:
        print(f"Error in predict_age: {e}")
        return f"Prediction Error: {str(e)}", "0%"

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
with gr.Blocks() as demo:
    gr.Markdown("# Speech Development Assessment Tool")
    
    with gr.Tabs():
        with gr.TabItem("Age Prediction", id="prediction"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(type="filepath", label="Upload Child's Voice Recording")
                    predict_btn = gr.Button("Predict Age Category")
                
                with gr.Column():
                    age_output = gr.Textbox(label="Predicted Age Category")
                    confidence_output = gr.Textbox(label="Prediction Confidence")
            
            predict_btn.click(
                fn=predict_age, 
                inputs=audio_input, 
                outputs=[age_output, confidence_output]
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
