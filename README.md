Hand and Face Recognition System with Sign Language Support
This project uses MediaPipe for hand and face detection and a machine learning model for hand gesture recognition. 
It integrates a subset of sign language gestures, enabling users to recognize and translate basic sign language into text and speech.pip install mediapipe


Features
Hand Gesture Recognition: Recognizes hand gestures and maps them to predefined labels.
Sign Language Translation: Supports basic sign language gestures and converts them into text and speech.
Face Detection: Detects faces and ensures accurate differentiation between hands and faces.
Text-to-Speech (TTS): Converts recognized gestures into spoken words.
Real-Time Processing: Captures video frames and processes gestures and faces in real-time.






Installation
Prerequisites
Python 3.8 or later
Required Libraries: Install these using the commands below:

pip install opencv-python

pip install numpy

pip install pyttsx3

pip install scikit-learn







Project Structure
project

├── model.p           # Pickled machine learning model for gesture recognition

├── model.py           # Main Python script

├── README.md         # Project documentation





How to Run
Clone this repository or download the code files.

Ensure your webcam is connected or built-in.

Run the main script:

bash
python model.py
Press q to quit the application.


How It Works
Gesture Recognition:

Captures landmarks of hand gestures using MediaPipe.
Uses a pre-trained machine learning model (model.p) to classify gestures.
Outputs the gesture label along with confidence percentage.
Sign Language Translation:

Recognizes gestures corresponding to a subset of sign language alphabets and numbers.
Displays the translated character and speaks it aloud if confidence is above 70%.
Face Detection:

Identifies faces in the video frame.
Prevents misclassification of hand gestures as faces.
Text-to-Speech (TTS):

Converts recognized gestures into spoken words or characters.
Avoids repeating the same word continuously.
Supported Sign Language Gestures
The system currently supports a subset of American Sign Language (ASL) alphabets and digits, including:

Alphabets: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
Numbers: 0-9


Known Issues
Hand and face overlap can occasionally cause false positives.
The TTS feature may lag if gestures are rapidly changing.
Future Enhancements
Expanding the system to support additional sign language gestures.
Fine-tuning face and hand detection to improve accuracy.
Adding multi-language support for TTS.
Incorporating real-time sentence formation for sign language phrases.
