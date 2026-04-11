"""
Twilio WhatsApp integration for field officer alerts.
Sends localized alerts to officers in the field.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from twilio.rest import Client as TwilioClient
    HAS_TWILIO = True
except ImportError:
    HAS_TWILIO = False
    logger.info("Twilio not installed. WhatsApp alerts run in mock mode.")

from config.settings import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER


class WhatsAppNotifier:
    """Send WhatsApp alerts to field officers via Twilio."""

    def __init__(
        self,
        account_sid: str = None,
        auth_token: str = None,
        from_number: str = None,
    ):
        self.from_number = from_number or TWILIO_WHATSAPP_NUMBER
        self._client = None
        self._mock_mode = True

        sid = account_sid or TWILIO_ACCOUNT_SID
        token = auth_token or TWILIO_AUTH_TOKEN

        if HAS_TWILIO and sid and token:
            try:
                self._client = TwilioClient(sid, token)
                self._mock_mode = False
                logger.info("Twilio WhatsApp client initialized")
            except Exception as e:
                logger.warning(f"Twilio init failed: {e}. Running in mock mode.")
        else:
            logger.info("WhatsApp notifications in mock mode")

    def send_alert(
        self,
        to_number: str,
        detection: Dict,
        language: str = "en",
    ) -> Dict:
        """
        Send a WhatsApp alert to a field officer.

        Args:
            to_number: Officer's phone number (format: "+91XXXXXXXXXX")
            detection: Detection data dict
            language: Message language ("en", "hi", "gu")

        Returns:
            {"success": bool, "message_sid": str, "body": str}
        """
        body = self._format_alert(detection, language)
        whatsapp_to = f"whatsapp:{to_number}" if not to_number.startswith("whatsapp:") else to_number

        if self._mock_mode:
            logger.info(f"[MOCK WhatsApp] To: {whatsapp_to}\n{body}")
            return {"success": True, "message_sid": "MOCK_SID_001", "body": body}

        try:
            message = self._client.messages.create(
                body=body,
                from_=self.from_number,
                to=whatsapp_to,
            )
            logger.info(f"WhatsApp sent: {message.sid} to {to_number}")
            return {"success": True, "message_sid": message.sid, "body": body}
        except Exception as e:
            logger.error(f"WhatsApp send failed: {e}")
            return {"success": False, "message_sid": None, "body": body, "error": str(e)}

    def _format_alert(self, detection: Dict, language: str) -> str:
        """Format the alert message in the specified language."""
        lat = detection.get("lat", 0)
        lon = detection.get("lon", 0)
        ct = detection.get("change_type", "unknown").replace("_", " ")
        conf = detection.get("confidence", 0) * 100
        area = detection.get("area_hectares", 0)
        maps_link = f"https://maps.google.com/?q={lat},{lon}"
        det_id = str(detection.get("id", ""))[:8]

        if language == "gu":
            return (
                f"🛰️ *સુરત મ્યુનિસિપલ કોર્પોરેશન — તપાસ ચેતવણી*\n\n"
                f"નવી તપાસ: ઝોનમાં *{ct}* ઓળખાયું\n"
                f"📍 સ્થાન: {lat:.4f}°N, {lon:.4f}°E\n"
                f"📐 વિસ્તાર: {area:.2f} હેક્ટર\n"
                f"💯 આત્મવિશ્વાસ: {conf:.0f}%\n"
                f"🗺️ નકશો: {maps_link}\n\n"
                f"કૃપા કરીને સ્થળની મુલાકાત લો અને તપાસ કરો.\n"
                f"ID: {det_id}"
            )
        elif language == "hi":
            return (
                f"🛰️ *सूरत नगर निगम — तपास चेतावनी*\n\n"
                f"नई पहचान: ज़ोन में *{ct}* पाया गया\n"
                f"📍 स्थान: {lat:.4f}°N, {lon:.4f}°E\n"
                f"📐 क्षेत्रफल: {area:.2f} हेक्टेयर\n"
                f"💯 आत्मविश्वास: {conf:.0f}%\n"
                f"🗺️ मानचित्र: {maps_link}\n\n"
                f"कृपया स्थल का दौरा करें और जाँच करें.\n"
                f"ID: {det_id}"
            )
        else:
            return (
                f"🛰️ *Surat Municipal Corporation — Field Alert*\n\n"
                f"New detection: *{ct}* identified in zone\n"
                f"📍 Location: {lat:.4f}°N, {lon:.4f}°E\n"
                f"📐 Area: {area:.2f} hectares\n"
                f"💯 Confidence: {conf:.0f}%\n"
                f"🗺️ Map: {maps_link}\n\n"
                f"Please visit the site and verify.\n"
                f"ID: {det_id}"
            )

    def dispatch_officer(
        self,
        officer_phone: str,
        detection: Dict,
        language: str = "gu",
        report_url: Optional[str] = None,
    ) -> Dict:
        """
        Full officer dispatch: send alert + location + report link.
        """
        if report_url:
            detection = {**detection, "report_url": report_url}

        result = self.send_alert(officer_phone, detection, language)

        # If report URL provided, send as follow-up
        if report_url and result["success"]:
            follow_up = f"📄 Report: {report_url}"
            if not self._mock_mode:
                try:
                    to = f"whatsapp:{officer_phone}" if not officer_phone.startswith("whatsapp:") else officer_phone
                    self._client.messages.create(
                        body=follow_up, from_=self.from_number, to=to,
                    )
                except Exception as e:
                    logger.warning(f"Follow-up send failed: {e}")

        return result
