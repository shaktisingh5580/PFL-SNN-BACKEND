import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class WahaClient:
    def __init__(self, base_url: str = "http://localhost:3000", session_name: str = "default"):
        self.base_url = base_url
        self.session_name = session_name
    
    def send_message(self, phone_number: str, message: str) -> bool:
        """Sends a plain text WhatsApp message via WAHA."""
        try:
            # Clean non-digits from phone number
            clean_number = "".join(filter(str.isdigit, str(phone_number)))
            if not clean_number:
                # Fallback to .env target if provided number is invalid
                clean_number = "".join(filter(str.isdigit, str(os.getenv("WHATSAPP_TARGET", ""))))
            
            if not clean_number:
                logger.error("WAHA: No valid phone number provided.")
                return False
                
            chat_id = f"{clean_number}@c.us"
            
            payload = {"chatId": chat_id, "text": message, "session": self.session_name}
            headers = {"Content-Type": "application/json", "accept": "application/json"}
            api_key = os.getenv("WAHA_API_KEY")
            if api_key: headers["X-Api-Key"] = api_key
            
            response = requests.post(f"{self.base_url}/api/sendText", json=payload, headers=headers, timeout=10)
            
            if response.status_code in (200, 201):
                logger.info(f"WhatsApp message dispatched to {chat_id}")
                return True
            else:
                logger.error(f"WAHA HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"WAHA dispatch failed: {e}")
            return False

# Global instance
waha = WahaClient()
