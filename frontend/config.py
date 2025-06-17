class Config:
    BACKEND_URL = "http://localhost:8000"
    MODERATE_ENDPOINT = f"{BACKEND_URL}/moderate"
    TRANSCRIBE_ENDPOINT = f"{BACKEND_URL}/transcribe-audio"
    REQUEST_TIMEOUT = 120
    
    MAX_FILE_SIZE_MB = 10
    ALLOWED_AUDIO_FORMATS = ['wav', 'mp3', 'ogg', 'm4a']
