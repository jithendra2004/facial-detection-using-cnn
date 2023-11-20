## MUSIC THERAPY USING EMOTION DETECTION USING REAL TIME CAPTURE USING CNN ALGORITHM
AIM:The primary goal of this project is to create an innovative system that seamlessly integrates real-time emotion detection using CNNs into music therapy sessions.
## FEATURES
1.Real-Time Emotion Detection

2.Data Collection Interface

3.CNN Model Architecture

4.Integration with Music Player
## REQUIREMENTS:
1.jupyter notebook

2.opencv

3.numpy 

4.pygame
## FLOWCHART
![WhatsApp Image 2023-11-18 at 11 59 50_a83cdcbf](https://github.com/jithendra2004/facial-detection-using-cnn/assets/94226297/157d5af8-7fc5-47d8-a8cd-a2ae383cac75)
## PACKAGES
1.install opencv
2.install numpy
## APPLICATIONS OR USES
1.Healthcare and Mental Wellness:

Therapeutic Intervention: Implementation in hospitals, clinics, or therapy centers to assist therapists in customizing music therapy sessions based on patients' real-time emotional responses.
Mental Health: Aid individuals dealing with stress, anxiety, or mood disorders by providing personalized music selections that align with their emotional states.
## PROGRAM:
## EMOTION ANALYSER:
~~~

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load your custom image from your file system
path_to_your_image = 'C:/Users/JITHENDRA/Downloads/facial rec mini proj/images/captured_image.jpg'
custom_image = Image.open(path_to_your_image)

# Convert the image to grayscale 
custom_image = custom_image.convert('L')

# Resize the image to match the dimensions of the training images (48x48)
custom_image = custom_image.resize((48, 48))

# Convert the image to a numpy array
custom_image_array = np.array(custom_image)

# Normalize the pixel values
custom_image_array = custom_image_array.astype('float32') / 255.0

# Reshape the image to fit the model input shape
custom_image_array = custom_image_array.reshape((1, 48, 48, 1))

# Use the model to predict the emotion for this image
predicted_emotion = model.predict(custom_image_array)

# Define emotions dictionary 
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Plot the custom image and its predicted emotion
fig, ax = plt.subplots(figsize=(6, 6))

# Display the custom image
ax.imshow(custom_image_array.reshape(48, 48), cmap='gray')
ax.set_title(f'Predicted Emotion: {emotions[np.argmax(predicted_emotion)]}')
ax.axis('off')

plt.show()


~~~
## PLAY MUSIC ACCORDING TO AN EMOTION
~~~
import pygame
import os

# Initialize pygame
pygame.init()

# Path to the folder containing music files
music_folder = r"C:\Users\JITHENDRA\Downloads\facial rec mini proj\pyhton\Songs"

# Get a list of music files in the emotion folder
music_files = [file for file in os.listdir(os.path.join(music_folder, emotion)) if file.endswith(".mp3")]

# Create a dictionary to map keys to music files
music_dict = {str(i + 1): os.path.join(music_folder, emotion, file) for i, file in enumerate(music_files)}

# Play songs continuously from the specified folder
try:
    for key in music_dict:
        pygame.mixer.music.load(music_dict[key])
        pygame.mixer.music.play(-1)  # -1 makes the music play indefinitely

        while pygame.mixer.music.get_busy():
            user_input = input("Press 's' to stop the music: ")
            if user_input.lower() == 's':
                pygame.mixer.music.stop()
                break
except KeyboardInterrupt:
    pygame.mixer.music.stop()
    pygame.quit()
~~~

## OUTPUT:
## IMAGE DETECTION
![image](https://github.com/jithendra2004/facial-detection-using-cnn/assets/94226297/77b54b18-6683-40de-81ab-3b047518e15a)
## ACCURACY OF THE PREDICTED IMAGE
![image](https://github.com/jithendra2004/facial-detection-using-cnn/assets/94226297/d2b48221-bc84-4d87-ace6-25ce32d5c8ac)
## PLAYING MUSIC
![image](https://github.com/jithendra2004/facial-detection-using-cnn/assets/94226297/b616446e-e7fe-4390-861c-cf8c4055ed81)

## RESULT:
Music therapy with real-time emotion detection through CNNs signifies a revolutionary leap in personalized therapeutic interventions. By harnessing technology to interpret and adapt to individual emotional responses during music therapy sessions, it unlocks new dimensions in tailored emotional support. This innovative fusion allows for dynamic adjustments in therapeutic approaches based on detected emotions, promising a more nuanced and personalized therapeutic experience.


