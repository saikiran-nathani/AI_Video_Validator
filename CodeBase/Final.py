import torch
import moviepy.editor as mp
from nltk.sentiment import SentimentIntensityAnalyzer
import whisper
import pytesseract
from PIL import Image

# Load the models
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base").to(device)
sentiment_analyzer = SentimentIntensityAnalyzer()
print(device)

def validate_video(video_path):
    """Validate video based on audio content and visual frames."""
    # Process the video to extract audio
    print("Processing video...")
    video = mp.VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    video.audio.write_audiofile(audio_path)

    # Transcribe the audio
    print("Transcribing audio...")
    audio_transcription = transcribe_audio(audio_path)

    # Analyze sentiment
    sentiment_scores = analyze_sentiment(audio_transcription)
    print("Audio sentiment analysis result:", sentiment_scores)

    # Check for code presence in the video
    frames = extract_frames_from_video(video)  # Extract frames for analysis
    visual_code_content = ""
    contains_code = any(extract_code_from_frame(frame) for frame in frames)

    # Predict topic from audio transcription
    predicted_topic = predict_topic(audio_transcription)

    # If no code is found, we still check if the transcript is relevant to programming
    is_relevant = is_relevant_topic(audio_transcription)

    # Check audio-video synchronization
    sync_result = check_audio_video_sync(audio_transcription, frames)
    print("Audio and video synchronization result:", sync_result)

    # Validation logic
    if contains_code:
        if sync_result:
            print(f"Predicted topic: {predicted_topic}")
            print("Validation result: pass (code and audio match)")
            return "pass"
        else:
            print(f"Predicted topic: {predicted_topic}")
            print("Validation result: reject (audio and video do not match)")
            return "reject"
    else:
        if is_relevant or sync_result:  # Allow explanatory videos to pass
            print(f"Predicted topic: {predicted_topic}")
            print("Validation result: pass (explanatory video or relevant audio)")
            return "pass"
        else:
            print(f"Predicted topic: {predicted_topic}")
            print("Validation result: reject (not relevant)")
            return "reject"


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    result = whisper_model.transcribe(audio_path, language="en")
    return result['text']


def analyze_sentiment(transcribed_text):
    """Perform sentiment analysis on the transcribed text."""
    scores = sentiment_analyzer.polarity_scores(transcribed_text)
    return scores


def predict_topic(transcribed_text):
    """Predict the main topic of the video based on the transcribed audio."""
    programming_topics = {
        "python": ["python", "flask", "django", "data science", "machine learning"],
        "java": ["spring", "jdk", "jre", "object-oriented"],
        "c#": ["c#", ".net", "asp.net"],
        "javascript": ["javascript", "node.js", "react", "vue.js", "js", "Javascript", "JavaScript"]
    }

    for topic, keywords in programming_topics.items():
        if any(keyword in transcribed_text.lower() for keyword in keywords):
            return topic

    return "Unknown topic"


def is_relevant_topic(transcribed_text):
    """Check if the transcribed text is relevant to programming topics."""
    relevant_terms = ["class", "object-oriented", "variable", "function", "method", "syntax", "code", "programming"]
    return any(term in transcribed_text.lower() for term in relevant_terms)


def extract_code_from_frame(frame):
    """Extract code or syntax from the video frame using OCR."""
    # Convert frame (NumPy array) to PIL Image for OCR processing
    image = Image.fromarray(frame)

    # Use pytesseract to perform OCR on the frame
    extracted_text = pytesseract.image_to_string(image)

    # Check if the extracted text contains any code-related syntax
    code_indicators = [';', '{', '}', '(', ')', 'def', 'class', 'if', 'else', 'for', 'while', '=', 'return']
    if any(indicator in extracted_text for indicator in code_indicators):
        return extracted_text  # Return the extracted text if code indicators are found
    return ""  # Return an empty string if no code indicators are found


def extract_frames_from_video(video, interval=1):
    """Extract frames from the video at a regular interval."""
    frames = []
    for i, frame in enumerate(video.iter_frames()):
        # Extract one frame per `interval` seconds
        if i % (video.fps * interval) == 0:
            frames.append(frame)
    return frames


def check_audio_video_sync(transcribed_text, frames):
    """Check if the audio and video content are synchronized."""
    # Collect the visual code content from each frame
    visual_code_content = " ".join(extract_code_from_frame(frame) for frame in frames)

    # Check if the transcribed audio content is relevant or found in the extracted code content
    return transcribed_text.lower() in visual_code_content.lower() or is_relevant_topic(transcribed_text)


# Example usage
video_file_path = "C:/Users/Sai_Kiran/Downloads/Introduction to Python Programming _ Python for Beginners #lec1.mp4"
validate_video(video_file_path)
