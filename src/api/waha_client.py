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
        """
        Sends a WhatsApp message via WAHA Docker container.
        phone_number should be strictly digits with country code (e.g. 919876543210 for India)
        """
        try:
            # Strip non-digits from phone number
            clean_number = "".join(filter(str.isdigit, str(phone_number)))
            if not clean_number:
                logger.error("WAHA Send Failed: Invalid phone number")
                return False
                
            chat_id = f"{clean_number}@c.us"
            
            payload = {
                "chatId": chat_id,
                "text": message,
                "session": self.session_name
            }
            
            headers = {
                "Content-Type": "application/json",
                "accept": "application/json",
            }
            api_key = os.getenv("WAHA_API_KEY")
            if api_key:
                headers["X-Api-Key"] = api_key
            
            response = requests.post(
                f"{self.base_url}/api/sendText",
                json=payload,
                headers=headers,
                timeout=5
            )
            
            if response.status_code in (200, 201):
                logger.info(f"WhatsApp Notification sent successfully to {chat_id}")
                return True
            else:
                logger.error(f"WAHA HTTP Error {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to WAHA Docker API: {e}")
            return False

# Global instance ready for import
waha = WahaClient()
