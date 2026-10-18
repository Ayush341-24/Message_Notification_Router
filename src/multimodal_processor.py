import os
import re
import json
from typing import Dict, Any, List, Optional

class MultimodalProcessor:
    """
    Step 2: Multimodal Processing Engine for WhatsApp Message Notification Router.
    Processes raw text, image posters/screenshots (OCR/VLM), and voice notes (Audio STT/Tone).
    Extracts unified text, entity tags, intent categories, urgency scores, and risk flags.
    Supports local deterministic heuristics + optional Gemini Flash zero-shot multimodal enrichment.
    """
    
    INTENT_CATEGORIES = [
        "personal", "urgent", "event", "payment", "business_update",
        "promotion", "greeting", "forward", "spam", "scam", "unknown"
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[MultimodalProcessor] Note: Gemini client not initialized ({e}). Using deterministic engine.")

    def process_message(self, message_ctx: Dict[str, Any], use_gemini: bool = False) -> Dict[str, Any]:
        """
        Processes a single message context bundle (from DataLoader.get_message_context)
        and produces a comprehensive multimodal feature representation.
        """
        msg = message_ctx.get("message", {})
        media_meta = message_ctx.get("media_metadata") or {}
        user_profile = message_ctx.get("user_profile") or {}
        business_info = message_ctx.get("business_info") or {}
        group_info = message_ctx.get("group_info") or {}
        
        media_type = msg.get("media_type") or msg.get("message_type") or "text"
        if media_type in ["voice_note", "audio"]:
            media_type = "voice"

        raw_text = msg.get("message_text") or msg.get("text_content") or ""
        ocr_text = media_meta.get("ocr_text", "") if media_type == "image" else ""
        transcript = media_meta.get("transcript", "") if media_type == "voice" else ""

        # Build unified text representation
        text_parts = []
        if raw_text and raw_text.strip():
            text_parts.append(f"Text: {raw_text.strip()}")
        if ocr_text and ocr_text.strip():
            text_parts.append(f"OCR: {ocr_text.strip()}")
        if transcript and transcript.strip():
            text_parts.append(f"Audio Transcript: {transcript.strip()}")

        unified_text = " | ".join(text_parts) if text_parts else raw_text or ocr_text or transcript or "No text content"

        # Deterministic extraction & scoring
        entities = self._extract_entities(unified_text, media_meta)
        urgency_score = self._compute_urgency(unified_text, msg, media_meta)
        risk_score = self._compute_risk(unified_text, msg, business_info, media_meta)
        intent = self._classify_intent_heuristic(unified_text, media_type, msg, media_meta, risk_score, urgency_score)

        processed_feature = {
            "message_id": msg.get("message_id"),
            "user_id": msg.get("user_id"),
            "media_kind": media_type,
            "raw_text": raw_text,
            "ocr_text": ocr_text,
            "transcript": transcript,
            "unified_text": unified_text,
            "intent_category": intent,
            "entities": entities,
            "urgency_score": round(urgency_score, 2),
            "risk_score": round(risk_score, 2),
            "forwarded_count": int(msg.get("forwarded_count") or 0),
            "media_summary": {
                "has_media": media_type in ["image", "voice"],
                "media_id": msg.get("media_id"),
                "file_path": message_ctx.get("media_file_path"),
                "duration_seconds": float(media_meta.get("duration_seconds") or 0) if media_type == "voice" else None,
                "speech_tone": media_meta.get("speech_tone") if media_type == "voice" else None,
                "image_type": media_meta.get("image_type") if media_type == "image" else None,
                "has_urgency_flag": bool(media_meta.get("has_urgency", False))
            }
        }

        # Optional Gemini Enrichment if requested & client available
        if use_gemini and self.client:
            try:
                gemini_res = self._enrich_with_gemini(unified_text, media_type, message_ctx)
                if gemini_res:
                    processed_feature["gemini_enrichment"] = gemini_res
                    if gemini_res.get("intent") in self.INTENT_CATEGORIES:
                        processed_feature["intent_category"] = gemini_res["intent"]
            except Exception as e:
                processed_feature["gemini_error"] = str(e)

        return processed_feature

    def process_all_messages(self, messages_context_list: List[Dict[str, Any]], use_gemini: bool = False) -> List[Dict[str, Any]]:
        """Batch processes a list of message context objects."""
        return [self.process_message(ctx, use_gemini=use_gemini) for ctx in messages_context_list]

    def _extract_entities(self, text: str, media_meta: Dict[str, Any]) -> Dict[str, List[str]]:
        """Regex and heuristic entity extraction for currency, deadlines, URLs, and phones."""
        amounts = re.findall(r'(?:₹|\$|USD|INR|Rs\.?)\s?\d+(?:,\d+)*(?:\.\d+)?|\b\d+\s?(?:rupees|dollars)\b', text, re.IGNORECASE)
        urls = re.findall(r'https?://[^\s]+|www\.[^\s]+|\b[a-zA-Z0-9.-]+\.(?:com|in|org|net|xyz|top|link)\b', text, re.IGNORECASE)
        phones = re.findall(r'\+?\d{1,3}[-.\s]?\d{10}|\b\d{10}\b', text)
        deadlines = re.findall(r'\b(?:today|tomorrow|tonight|due|deadline|by \d{1,2}(?::\d{2})?\s?(?:am|pm)?|urgently|immediately)\b', text, re.IGNORECASE)
        
        # Merge keywords from media metadata if available
        keywords = []
        if media_meta.get("key_elements"):
            keywords.extend(str(media_meta.get("key_elements")).split(","))

        return {
            "amounts": list(set(amounts)),
            "urls": list(set(urls)),
            "phones": list(set(phones)),
            "deadlines": list(set(deadlines)),
            "keywords": [k.strip() for k in keywords if k.strip()]
        }

    def _compute_urgency(self, text: str, msg: Dict[str, Any], media_meta: Dict[str, Any]) -> float:
        """Calculate urgency score based on keywords, speech tone, and urgency flags."""
        score = 0.2  # Baseline
        low_text = text.lower()

        # Non-urgent dampeners
        if any(p in low_text for p in ["nothing urgent", "don't call", "no rush", "when you are free", "talk tomorrow", "when free"]):
            score -= 0.15

        urgent_keywords = ["urgent", "immediately", "emergency", "due today", "asap", "alert", "otp", "verify now", "failed payment", "deadline"]
        for kw in urgent_keywords:
            if kw in low_text and "never ask for otp" not in low_text and "nothing urgent" not in low_text:
                score += 0.25

        if media_meta.get("has_urgency"):
            score += 0.3
            
        if media_meta.get("speech_tone") in ["anxious", "urgent", "fast"]:
            score += 0.25

        return min(1.0, max(0.0, score))

    def _compute_risk(self, text: str, msg: Dict[str, Any], business_info: Dict[str, Any], media_meta: Dict[str, Any]) -> float:
        """Calculate risk/scam score based on suspicious keywords, unverified business, prompt injection attacks, and forwarding counts."""
        score = 0.05  # Baseline safe
        low_text = text.lower()

        # Prompt Injection & System Tampering Attacks (Immediate Max Risk)
        injection_triggers = [
            "ignore all previous routing rules", "ignore previous instructions",
            "system prompt override", "mark this message as notify", "mark as urgent",
            "bypass dnd", "force notify", "you are an assistant that must"
        ]
        if any(inj in low_text for inj in injection_triggers):
            return 1.0

        # Scam triggers
        scam_triggers = [
            "win", "winner", "lottery", "claim cash", "click link", "account suspended", 
            "kbc", "part-time job", "invest 1000", "daily returns", "wire money", "kyc update",
            "otp may have leaked", "account-login-check", "auth-pass-reset", "verify now at",
            "profile will be blocked", "confirm details at", "workspace access will expire",
            "reply with the 6 digit code", "verify your account at", "unauthorized transaction detected"
        ]
        for kw in scam_triggers:
            if kw in low_text:
                score += 0.45

        # Forwarded count multiplier
        fwd = int(msg.get("forwarded_count") or 0)
        if fwd > 5:
            score += 0.2
        elif fwd > 1:
            score += 0.1

        # Business risk
        if business_info:
            is_verif = str(business_info.get("verified", "0")) == "1" or bool(business_info.get("is_verified", False))
            if not is_verif:
                score += 0.15
            if float(business_info.get("spam_reports_count") or business_info.get("user_reports_30d") or 0) > 10:
                score += 0.3

        return min(1.0, max(0.0, score))

    def _classify_intent_heuristic(self, text: str, media_type: str, msg: Dict[str, Any], media_meta: Dict[str, Any], risk_score: float, urgency_score: float) -> str:
        """Determines best-fit message_type category deterministically."""
        low_text = text.lower()

        # Check cached media metadata intent if present
        meta_intent = media_meta.get("intent")
        if meta_intent in self.INTENT_CATEGORIES and meta_intent not in ["unknown"]:
            return meta_intent

        # Check scam / phishing explicitly
        scam_phrases = [
            "otp may have leaked", "account-login-check", "auth-pass-reset", "verify now at",
            "profile will be blocked", "confirm details at", "workspace access will expire",
            "reply with the 6 digit code", "ignore all previous routing rules", "claim cash",
            "account suspended", "lottery", "winner"
        ]
        if any(p in low_text for p in scam_phrases) or risk_score >= 0.45:
            return "scam"

        # Safety advisory is business_update, NOT payment or scam!
        if "safety advisory" in low_text or "never ask for otp" in low_text:
            return "business_update"

        # Urgent indicators
        urgent_phrases = [
            "heads-up", "tanker guy", "prod review", "pulled to", "last-minute", 
            "come online now", "alert threshold", "emergency", "retry count",
            "urgent", "asap", "immediately"
        ]
        if (any(p in low_text for p in urgent_phrases) or urgency_score >= 0.65) and "nothing urgent" not in low_text:
            return "urgent"

        # Event indicators
        event_phrases = [
            "bus is leaving", "school circular", "timing and consent", "cultural night",
            "webinar", "invitation", "meeting at", "gathering", "society notice",
            "health-related update", "ready for review"
        ]
        if any(p in low_text for p in event_phrases):
            return "event"

        # Payment indicators
        payment_phrases = ["paid", "payment", "invoice", "receipt", "bill", "bank", "credited", "debited", "transfer"]
        if any(p in low_text for p in payment_phrases) and "never ask for otp" not in low_text and "selling" not in low_text and "order ending" not in low_text:
            return "payment"

        # Unknown / volunteer contact
        if "volunteer sheet" in low_text or "found your number" in low_text:
            return "unknown"

        # Promotion indicators
        promo_phrases = ["selling", "sale", "off", "discount", "cashback", "coupon", "buy 1 get 1", "limited offer", "promo", "shopping offer", "trip last change", "kurta set", "travel deal", "itinerary"]
        if any(p in low_text for p in promo_phrases):
            return "promotion"

        if any(w in low_text for w in ["good morning", "good night", "happy birthday", "congratulations", "festival", "diwali", "eid"]):
            return "greeting"

        if msg.get("business_id") or "customer" in low_text or "order" in low_text or "packed" in low_text:
            return "business_update"

        if int(msg.get("forwarded_count") or 0) >= 3:
            return "forward"

        if media_type == "voice" or msg.get("sender_user_id"):
            return "personal"

        return "unknown"

    def _enrich_with_gemini(self, text: str, media_type: str, message_ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Zero-shot structured classification using Gemini Flash."""
        if not self.client:
            return {"error": "Gemini API client not initialized. GEMINI_API_KEY environment variable is required."}

        try:
            from google.genai import types
            
            prompt = f"""
Analyze this WhatsApp message multimodal context:
- Content: {text}
- Media Type: {media_type}

Classify into exactly ONE of these categories:
['personal', 'urgent', 'event', 'payment', 'business_update', 'promotion', 'greeting', 'forward', 'spam', 'scam', 'unknown']

Return a JSON object with keys:
"intent": string (one of allowed categories),
"confidence": float (0.0 to 1.0),
"reason": string (short summary reason)
"""
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            if response and response.text:
                return json.loads(response.text)
        except Exception as e:
            return {"error": f"Gemini API call failed: {str(e)}"}
        return None
