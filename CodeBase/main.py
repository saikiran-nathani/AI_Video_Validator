import torch
import ffmpeg
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
    print("Processing video...")

    # Extract audio from video using ffmpeg
    audio_path = video_path.replace(".mp4", ".wav")
    extract_audio_ffmpeg(video_path, audio_path)

    # Transcribe the audio
    print("Transcribing audio...")
    audio_transcription = transcribe_audio(audio_path)

    # Analyze sentiment
    sentiment_scores = analyze_sentiment(audio_transcription)
    print("Audio sentiment analysis result:", sentiment_scores)

    # Extract frames from video using ffmpeg
    frames = extract_frames_from_video_ffmpeg(video_path)
    contains_code = any(extract_code_from_frame(frame) for frame in frames)

    # Predict topic from audio transcription
    predicted_topic = predict_topic(audio_transcription)

    # Check if the transcript is relevant to programming
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
        if is_relevant or sync_result:
            print(f"Predicted topic: {predicted_topic}")
            print("Validation result: pass (explanatory video or relevant audio)")
            return "pass"
        else:
            print(f"Predicted topic: {predicted_topic}")
            print("Validation result: reject (not relevant)")
            return "reject"

def extract_audio_ffmpeg(video_path, output_audio_path):
    """Extract audio using ffmpeg-python."""
    ffmpeg.input(video_path).output(output_audio_path).run(overwrite_output=True)

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
    image = Image.fromarray(frame)
    extracted_text = pytesseract.image_to_string(image)
    code_indicators = [';', '{', '}', '(', ')', 'def', 'class', 'if', 'else', 'for', 'while', '=', 'return']
    if any(indicator in extracted_text for indicator in code_indicators):
        return extracted_text
    return ""

def extract_frames_from_video_ffmpeg(video_path, interval=1):
    """Extract frames using ffmpeg + PIL."""
    import cv2
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    success, frame = vidcap.read()
    count = 0
    while success:
        if count % frame_interval == 0:
            frames.append(frame)
        success, frame = vidcap.read()
        count += 1
    vidcap.release()
    return frames

def check_audio_video_sync(transcribed_text, frames):
    """Check if the audio and video content are synchronized."""
    visual_code_content = " ".join(extract_code_from_frame(frame) for frame in frames)
    return transcribed_text.lower() in visual_code_content.lower() or is_relevant_topic(transcribed_text)

# Example usage
video_file_path = "C:\Users\Sai_Kiran\PycharmProjects\Artificial_Intelligence\Video_Content_Validator\Test_cases\Nallanchu Thellacheera Lyrical _ Mr Bachchan_ Ravi Teja _ Harish Shankar_ Bhagyashri _Mickey J Meyer.mp4"
validate_video(video_file_path)