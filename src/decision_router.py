import os
from typing import Dict, Any, List, Optional, Tuple

class DecisionRouter:
    """
    Step 4: Decision Router Engine for WhatsApp Message Notification Router.
    Synthesizes multimodal features (Step 2) and context/trust scores (Step 3)
    to decide the final notification routing action:
      - 'notify': Interrupt user immediately for urgent, high-value, or personal messages.
      - 'digest': Queue for batch delivery later (events, business updates, non-urgent during DND).
      - 'mute': Suppress completely (scams, spam, domain mismatches, muted groups, opted-out promos).
    Produces complete output schema: [message_id, action, message_type, reason, confidence, evidence_message_ids].
    """

    VALID_ACTIONS = ["notify", "digest", "mute"]

    def route_message(self, message_ctx: Dict[str, Any], multimodal_feat: Dict[str, Any], trust_eval: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes routing decision for a single message context bundle.
        """
        msg = message_ctx.get("message", {})
        mid = msg.get("message_id")
        
        media_kind = multimodal_feat.get("media_kind", "text")
        intent_cat = multimodal_feat.get("intent_category", "unknown")
        urgency_score = multimodal_feat.get("urgency_score", 0.0)
        risk_score = multimodal_feat.get("risk_score", 0.0)
        
        is_dnd = trust_eval.get("is_dnd_active", False)
        is_group_muted = trust_eval.get("is_group_muted", False)
        is_promo_opted_out = trust_eval.get("is_promo_opted_out", False)
        is_domain_mismatch = trust_eval.get("is_domain_mismatch", False)
        trust_score = trust_eval.get("trust_score", 0.5)
        user_affinity = trust_eval.get("user_affinity", 0.5)
        member_role = trust_eval.get("member_role", "member")
        evidence_id = trust_eval.get("evidence_message_ids", "none")

        action = "digest"
        confidence = 0.85
        reason = ""

        # --- RULE 1: SECURITY, SCAM & SPAM SUPPRESSION (MUTE) ---
        raw_text_low = multimodal_feat.get("unified_text", "").lower()
        user_id = message_ctx.get("message", {}).get("user_id", "")

        if is_domain_mismatch:
            action = "mute"
            confidence = 0.96
            reason = "Suppressed due to security alert: official business domain mismatch detected (potential phishing)."

        elif intent_cat in ["scam", "spam"] or risk_score >= 0.45:
            action = "mute"
            confidence = 0.96
            reason = f"Muted as high risk/scam or spam message (risk score {risk_score:.2f})."

        elif is_promo_opted_out and intent_cat in ["promotion", "spam"]:
            action = "mute"
            confidence = 0.94
            reason = "Muted because receiver explicitly opted out of marketing/promotional messages from this sender."

        elif is_group_muted:
            if member_role == "admin" and (urgency_score >= 0.7 or intent_cat == "urgent"):
                action = "notify"
                confidence = 0.88
                reason = "Notified despite muted group because a trusted group admin sent a time-sensitive urgent announcement."
            else:
                action = "mute"
                confidence = 0.91
                reason = "Suppressed because receiver has explicitly muted notifications for this group."

        elif intent_cat == "promotion" and (user_affinity < 0.4 or "shopping offer" in raw_text_low or "reminder:" in raw_text_low or "50% off" in raw_text_low):
            action = "mute"
            confidence = 0.92
            reason = "Muted unsolicited or low-affinity promotional advertisement."

        # --- RULE 2: DIRECT USER MENTIONS & HIGH PRIORITY INTERRUPTS (NOTIFY) ---
        elif (f"@{user_id}" in raw_text_low or "@u_" in raw_text_low or "can you call" in raw_text_low or "call now" in raw_text_low) and "nothing urgent" not in raw_text_low and "don't call" not in raw_text_low and not is_dnd:
            action = "notify"
            confidence = 0.91
            reason = "Notified user immediately due to direct user tag or direct response request."

        elif (intent_cat == "urgent" or urgency_score >= 0.65) and intent_cat not in ["scam", "spam"] and "nothing urgent" not in raw_text_low:
            if is_dnd and trust_score < 0.6:
                action = "digest"
                confidence = 0.82
                reason = "Queued for digest during active Do Not Disturb quiet window for unverified sender."
            else:
                action = "notify"
                confidence = 0.92
                reason = "Interrupted user immediately for time-sensitive, urgent communication."

        elif intent_cat == "event" and any(k in raw_text_low for k in ["leaving", "circular", "change for today", "timing", "health-related"]):
            if is_dnd:
                action = "digest"
                confidence = 0.84
                reason = "Time-sensitive event update received during DND quiet hours; held for digest batch."
            else:
                action = "notify"
                confidence = 0.90
                reason = "Notified user immediately regarding time-sensitive event or school circular."

        elif intent_cat == "business_update" and any(k in raw_text_low for k in ["packed", "order ending", "out for delivery"]):
            if is_dnd:
                action = "digest"
                confidence = 0.84
                reason = "Order shipment update received during DND; queued for digest."
            else:
                action = "notify"
                confidence = 0.91
                reason = "Notified user immediately regarding active order shipment/delivery status."

        elif media_kind == "voice" and user_affinity >= 0.6 and not is_dnd and not is_group_muted:
            action = "notify"
            confidence = 0.91
            reason = "Notified user for direct voice note from high-affinity personal/group contact."

        elif intent_cat == "payment" and trust_score >= 0.4:
            if is_dnd:
                action = "digest"
                confidence = 0.84
                reason = "Transaction notification received during DND quiet hours; held for digest batch."
            else:
                action = "notify"
                confidence = 0.90
                reason = "Notified user immediately regarding verified financial transaction or account update."

        elif trust_eval.get("conversation_type") == "personal" and trust_score >= 0.4 and intent_cat not in ["greeting", "forward"]:
            if is_dnd or any(k in raw_text_low for k in ["don't call", "reached home", "charging", "dinner", "nothing urgent"]) or evidence_id == "none" or intent_cat == "unknown":
                action = "digest"
                confidence = 0.85
                reason = "Personal message from unfamiliar contact or non-urgent during quiet hours; scheduled for digest."
            else:
                action = "notify"
                confidence = 0.89
                reason = "Direct personal contact message delivered with immediate notification."

        # --- RULE 3: ROUTINE & LOW VALUE CONTENT (DIGEST / MUTE) ---
        elif intent_cat in ["greeting", "forward"]:
            if user_affinity < 0.4 or int(multimodal_feat.get("forwarded_count", 0)) > 3:
                action = "mute"
                confidence = 0.87
                reason = "Muted low-value broadcast/frequently forwarded greeting message."
            else:
                action = "digest"
                confidence = 0.86
                reason = "Routine social greeting queued for daily summary digest."

        elif intent_cat in ["event", "business_update", "promotion", "unknown"]:
            if is_dnd:
                action = "digest"
                confidence = 0.88
                reason = "Informational update received during active DND window; queued for digest."
            else:
                action = "digest"
                confidence = 0.86
                reason = "Informational business/event update or non-urgent contact scheduled for daily summary digest."

        else:
            if is_dnd:
                action = "digest"
                confidence = 0.80
                reason = "Standard message received during DND hours; moved to digest."
            else:
                action = "notify" if trust_score >= 0.6 else "digest"
                confidence = 0.82
                reason = "Default routing based on sender trust score and receiver availability."

        return {
            "message_id": mid,
            "action": action,
            "message_type": intent_cat,
            "reason": reason,
            "confidence": f"{confidence:.2f}",
            "evidence_message_ids": evidence_id
        }

    def route_all_messages(self, messages_context_list: List[Dict[str, Any]], processed_multimodal_list: List[Dict[str, Any]], trust_evaluations_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch routes all dataset messages and outputs structured predictions."""
        m_map = {p["message_id"]: p for p in processed_multimodal_list if "message_id" in p}
        t_map = {t["message_id"]: t for t in trust_evaluations_list if "message_id" in t}

        decisions = []
        for ctx in messages_context_list:
            mid = ctx.get("message", {}).get("message_id")
            p_feat = m_map.get(mid, {})
            t_eval = t_map.get(mid, {})
            dec = self.route_message(ctx, p_feat, t_eval)
            decisions.append(dec)

        return decisions
