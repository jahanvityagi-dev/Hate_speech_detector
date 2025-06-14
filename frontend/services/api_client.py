import requests
from config import Config
from core.models import ModerationResult
from typing import Dict, Any, Optional, Tuple, List
import json
class APIClient:
    """Centralized API client for backend communication."""
    
    @staticmethod
    def _make_request(endpoint: str, data: Dict[str, Any] = None, 
                     files: Dict = None) -> Dict[str, Any]:
        """
        Make API request with comprehensive error handling.
        
        Args:
            endpoint: API endpoint URL
            data: JSON data for POST request
            files: Files for multipart upload
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.RequestException: For various request errors
        """
        try:
            if files:
                response = requests.post(
                    endpoint, 
                    files=files, 
                    timeout=Config.REQUEST_TIMEOUT
                )
            else:
                response = requests.post(
                    endpoint, 
                    json=data, 
                    timeout=Config.REQUEST_TIMEOUT
                )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            raise requests.RequestException(
                "Cannot connect to backend server. "
                f"Please ensure the API is running at {Config.BACKEND_URL}"
            )
        except requests.exceptions.Timeout:
            raise requests.RequestException("Request timed out. Please try again.")
        except requests.exceptions.HTTPError as e:
            raise requests.RequestException(
                f"API Error: {e.response.status_code} - {e.response.text}"
            )
        except json.JSONDecodeError:
            raise requests.RequestException("Invalid response from server.")
    
    @classmethod
    def moderate_text(cls, text: str) -> ModerationResult:
        """
        Send text for moderation analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            ModerationResult object
        """
        response = cls._make_request(Config.MODERATE_ENDPOINT, {"text": text})
        return ModerationResult(
            label=response.get('label', 'Unknown'),
            classification_reason=response.get('classification_reason', ''),
            explanation=response.get('explanation', ''),
            policy_snippets=response.get('policy_snippets', []),
            action=response.get('action', ''),
            action_reason=response.get('action_reason', ''),
            input_text=text
        )
    
    @classmethod
    def transcribe_and_moderate(cls, audio_file) -> Tuple[ModerationResult, str]:
        """
        Send audio for transcription and moderation.
        
        Args:
            audio_file: Audio file object
            
        Returns:
            Tuple of (ModerationResult, transcription)
        """
        files = {"file": (audio_file.name, audio_file.getvalue(), "audio/wav")}
        response = cls._make_request(Config.TRANSCRIBE_ENDPOINT, files=files)
        
        transcription = response.get('transcribed_text', '')
        moderation_data = response.get('moderation_result', {})
        
        moderation_result = ModerationResult(
            label=moderation_data.get('label', 'Unknown'),
            classification_reason=moderation_data.get('classification_reason', ''),
            explanation=moderation_data.get('explanation', ''),
            policy_snippets=moderation_data.get('policy_snippets', []),
            action=moderation_data.get('action', ''),
            action_reason=moderation_data.get('action_reason', ''),
            input_text=transcription
        )
        
        return moderation_result, transcription
