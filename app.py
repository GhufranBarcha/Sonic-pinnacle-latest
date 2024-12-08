import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gradio as gr

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

def audio_to_spectrogram(audio_path, save_path):
    """Convert audio file to spectrogram image."""
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

def predict_age(audio_file):
    """Predict age based on audio input."""
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

# Create Gradio interface
iface = gr.Interface(
    fn=predict_age,
    inputs=gr.Audio(type="filepath", label="Upload Child's Voice Recording"),
    outputs=[
        gr.Textbox(label="Predicted Age Category"),
        gr.Textbox(label="Prediction Confidence")
    ],
    title="Child Speech Age Development Predictor",
    description="Upload a .wav audio file of a child's voice to predict their speech development category.",
    allow_flagging="never"
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()