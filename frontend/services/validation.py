from config import Config
from pathlib import Path
from typing import Union, IO
from pathlib import Path


class ValidationError(Exception):
    pass

class TextValidator:
    """Validation utilities for user input."""

    @staticmethod
    def validate_text(text: str) -> bool:
        """Validate that text is non-empty and meets minimum length."""
        return text is not None and text.strip() != "" and len(text.strip()) >= 3

class AudioValidator:
    @staticmethod
    def validate(file):
        if file is None:
            raise ValidationError("No audio file")

        ext = Path(file.name).suffix.lower().lstrip('.')
        if ext not in Config.ALLOWED_AUDIO_FORMATS:
            raise ValidationError("Unsupported format")

        size_mb = file.size / (1024 * 1024)
        if size_mb > Config.MAX_FILE_SIZE_MB:
            raise ValidationError("File too large")

    @staticmethod
    def validate_audio_file(file: Union[IO, any]):
        # Use `name` if available, else skip extension check
        if not hasattr(file, 'name'):
            return  # Skip extension check for in-memory audio

        ext = Path(file.name).suffix.lower().lstrip('.')
        if ext not in Config.ALLOWED_AUDIO_FORMATS:
            raise ValidationError("Unsupported audio format")

        # Only check size if `size` attribute exists
        if hasattr(file, "size"):
            size_mb = file.size / (1024 * 1024)
            if size_mb > Config.MAX_FILE_SIZE_MB:
                raise ValidationError("File too large")
