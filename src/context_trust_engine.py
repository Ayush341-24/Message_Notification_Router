import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

class ContextTrustEngine:
    """
    Step 3: Context & Trust Scoring Engine for WhatsApp Message Notification Router.
    Computes receiver quiet hours (DND), sender trust scores, domain verification flags,
    group channel mute statuses, user affinity, and retrieves evidence historical message IDs.
    """

    def __init__(self, data_loader: Any):
        self.loader = data_loader

    def evaluate_context_and_trust(self, message_ctx: Dict[str, Any], processed_multimodal: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evaluates full context & trust metrics for a message given its raw context and Step 2 features.
        Supports optional weight overrides for interactive policy testing.
        """
        msg = message_ctx.get("message", {})
        user_profile = message_ctx.get("user_profile") or {}
        group_info = message_ctx.get("group_info") or {}
        business_info = message_ctx.get("business_info") or {}
        user_biz_hist = message_ctx.get("user_business_history") or {}
        
        user_id = msg.get("user_id")
        created_at_str = msg.get("created_at", "")
        conversation_type = msg.get("conversation_type") or self._infer_conversation_type(msg)

        # 1. Receiver Quiet Hours / DND Check
        is_dnd_active = self._check_dnd_window(user_profile.get("do_not_disturb_window", ""), created_at_str)

        # 2. Channel & Group Mute Check
        is_group_muted = False
        member_role = "member"
        if group_info:
            gid = group_info.get("group_id")
            group_members = self.loader.group_members.get(gid, [])
            user_member_record = next((m for m in group_members if m.get("user_id") == user_id), {})
            if user_member_record:
                is_group_muted = str(user_member_record.get("group_muted_by_user", "0")) == "1"
                member_role = user_member_record.get("role", "member")

        # 3. Sender Verification & Domain Mismatch Check
        is_domain_mismatch = False
        is_verified_business = False
        is_promo_opted_out = False
        
        if business_info:
            is_verified_business = str(business_info.get("verified", "0")) == "1"
            official_dom = str(business_info.get("official_domain", "")).lower().strip()
            sender_dom = str(business_info.get("domain_used_by_sender", "")).lower().strip()
            if not is_verified_business and official_dom and sender_dom and official_dom != sender_dom:
                is_domain_mismatch = True
            
            if user_biz_hist:
                allows_promo = str(user_biz_hist.get("allows_promotions", "1")) == "1"
                opted_out_at = str(user_biz_hist.get("promotions_opted_out_at", "")).strip()
                if not allows_promo or opted_out_at:
                    is_promo_opted_out = True

        # 4. Sender Trust Score Calculation (0.0 to 1.0)
        trust_score, trust_signals = self._calculate_trust_score(
            conversation_type=conversation_type,
            business_info=business_info,
            user_biz_hist=user_biz_hist,
            group_info=group_info,
            member_role=member_role,
            is_domain_mismatch=is_domain_mismatch,
            processed_multimodal=processed_multimodal,
            weights=weights
        )

        # 5. User Affinity & Historical Engagement Score (0.0 to 1.0)
        user_affinity = self._calculate_user_affinity(
            user_id=user_id,
            msg=msg,
            user_profile=user_profile,
            user_biz_hist=user_biz_hist,
            group_info=group_info
        )

        # 6. Evidence Message Retrieval
        evidence_ids = self._retrieve_evidence_message_ids(
            msg=msg,
            user_id=user_id,
            conversation_type=conversation_type,
            group_id=msg.get("group_id"),
            business_id=msg.get("business_id"),
            sender_user_id=msg.get("sender_user_id") or msg.get("sender_id")
        )

        return {
            "message_id": msg.get("message_id"),
            "user_id": user_id,
            "conversation_type": conversation_type,
            "is_dnd_active": is_dnd_active,
            "is_group_muted": is_group_muted,
            "is_promo_opted_out": is_promo_opted_out,
            "is_domain_mismatch": is_domain_mismatch,
            "is_verified_business": is_verified_business,
            "trust_score": round(trust_score, 2),
            "user_affinity": round(user_affinity, 2),
            "member_role": member_role,
            "evidence_message_ids": evidence_ids,
            "trust_signals": trust_signals,
            "summary_context": self._build_context_summary(
                is_dnd=is_dnd_active,
                is_group_muted=is_group_muted,
                is_domain_mismatch=is_domain_mismatch,
                is_verified=is_verified_business,
                trust_score=trust_score,
                evidence_ids=evidence_ids
            )
        }

    def evaluate_all_messages(self, messages_context_list: List[Dict[str, Any]], processed_multimodal_list: List[Dict[str, Any]], weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Batch evaluates context and trust for all dataset messages."""
        p_map = {p["message_id"]: p for p in processed_multimodal_list if "message_id" in p}
        results = []
        for ctx in messages_context_list:
            mid = ctx.get("message", {}).get("message_id")
            p_feat = p_map.get(mid, {})
            res = self.evaluate_context_and_trust(ctx, p_feat, weights=weights)
            results.append(res)
        return results

    def _infer_conversation_type(self, msg: Dict[str, Any]) -> str:
        if msg.get("group_id") and str(msg.get("group_id")).strip():
            return "group"
        if msg.get("business_id") and str(msg.get("business_id")).strip():
            return "business"
        return "personal"

    def _check_dnd_window(self, dnd_window_str: str, created_at_str: str) -> bool:
        """Parse DND window (e.g. '22:00-07:00') and compare with message timestamp."""
        if not dnd_window_str or "-" not in dnd_window_str or not created_at_str:
            return False
        
        try:
            # Extract time from 'YYYY-MM-DD HH:MM' or 'HH:MM'
            time_part = created_at_str.strip().split(" ")[-1] if " " in created_at_str else created_at_str
            msg_time = datetime.strptime(time_part, "%H:%M").time()

            start_str, end_str = dnd_window_str.split("-")
            start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_time = datetime.strptime(end_str.strip(), "%H:%M").time()

            if start_time <= end_time:
                return start_time <= msg_time <= end_time
            else: # Crosses midnight e.g. 22:00 to 07:00
                return msg_time >= start_time or msg_time <= end_time
        except Exception:
            return False

    def _calculate_trust_score(self, conversation_type: str, business_info: Dict[str, Any], user_biz_hist: Dict[str, Any], group_info: Dict[str, Any], member_role: str, is_domain_mismatch: bool, processed_multimodal: Dict[str, Any], weights: Optional[Dict[str, float]] = None) -> Tuple[float, List[str]]:
        w = weights or {}
        w_verified = w.get("verified_boost", 0.25)
        w_domain_mismatch = w.get("domain_mismatch_penalty", -0.45)
        w_admin = w.get("admin_boost", 0.25)
        
        score = 0.5  # Baseline neutral
        signals = []

        risk_score = processed_multimodal.get("risk_score", 0.0)

        if conversation_type == "business":
            if str(business_info.get("verified", "0")) == "1":
                score += w_verified
                signals.append(f"Verified Business Badge (+{round(w_verified, 2)})")
            else:
                score -= 0.1
                signals.append("Unverified Business Account")

            if is_domain_mismatch:
                score += w_domain_mismatch # w_domain_mismatch is negative e.g. -0.45
                signals.append(f"⚠️ Domain Mismatch Phishing Alert ({round(w_domain_mismatch, 2)})")

            acc_age = int(business_info.get("account_age_days") or 0)
            if acc_age > 365:
                score += 0.1
                signals.append("Established Account (> 1 Year)")
            elif acc_age < 30:
                score -= 0.2
                signals.append("New Business Account (< 30 Days)")

            reports = int(business_info.get("user_reports_30d") or 0)
            if reports > 10:
                score -= 0.3
                signals.append(f"High User Spam Reports ({reports})")

            if user_biz_hist:
                replied_30d = int(user_biz_hist.get("messages_replied_30d") or 0)
                if replied_30d > 0:
                    score += 0.2
                    signals.append(f"User Replied {replied_30d}x in 30d")

        elif conversation_type == "group":
            if member_role == "admin":
                score += w_admin
                signals.append(f"Group Admin Sender (+{round(w_admin, 2)})")
            else:
                score += 0.05
                signals.append("Regular Group Member")

            g_msgs = int(group_info.get("messages_30d") or 0)
            if g_msgs > 50:
                signals.append("High Activity Group")

        else:  # personal direct
            score += 0.2
            signals.append("Direct Personal Contact")

        # Deduct risk score penalty
        if risk_score > 0.4:
            score -= (risk_score * 0.4)
            signals.append(f"Multimodal Risk Penalty (-{round(risk_score * 0.4, 2)})")

        final_trust = min(1.0, max(0.0, score))
        return final_trust, signals

    def _calculate_user_affinity(self, user_id: str, msg: Dict[str, Any], user_profile: Dict[str, Any], user_biz_hist: Dict[str, Any], group_info: Dict[str, Any]) -> float:
        affinity = 0.5  # Baseline
        
        if user_biz_hist:
            opened = int(user_biz_hist.get("messages_opened_30d") or 0)
            replied = int(user_biz_hist.get("messages_replied_30d") or 0)
            dismissed = int(user_biz_hist.get("messages_dismissed_30d") or 0)

            if replied > 0:
                affinity += 0.3
            elif opened > 2:
                affinity += 0.15
            
            if dismissed > 3:
                affinity -= 0.2

        if group_info:
            gid = group_info.get("group_id")
            members = self.loader.group_members.get(gid, [])
            mem_rec = next((m for m in members if m.get("user_id") == user_id), {})
            if mem_rec:
                reads = int(mem_rec.get("messages_read_30d") or 0)
                replies = int(mem_rec.get("replies_sent_30d") or 0)
                if replies > 0 or reads > 10:
                    affinity += 0.2

        return min(1.0, max(0.0, affinity))

    def _retrieve_evidence_message_ids(self, msg: Dict[str, Any], user_id: str, conversation_type: str, group_id: Optional[str], business_id: Optional[str], sender_user_id: Optional[str]) -> str:
        """Finds evidence message IDs from message_history.csv / message_events.csv."""
        candidates = []
        
        for h_msg in self.loader.message_history:
            if h_msg.get("user_id") != user_id:
                continue

            h_id = h_msg.get("message_id")
            if not h_id:
                continue
            
            # Match by channel
            if conversation_type == "business" and business_id and (h_msg.get("business_id") == business_id or h_msg.get("sender_id") == business_id):
                candidates.append(h_id)
            elif conversation_type == "group" and group_id and h_msg.get("group_id") == group_id:
                candidates.append(h_id)
            elif conversation_type == "personal" and sender_user_id and (h_msg.get("sender_user_id") == sender_user_id or h_msg.get("sender_id") == sender_user_id):
                candidates.append(h_id)

        if not candidates:
            return "none"

        # Check message events for candidates to find highest engaged message
        prioritized = []
        for cid in candidates:
            events = self.loader.message_events.get(cid, [])
            if any(str(e.get("message_replied", "0")) == "1" or str(e.get("message_opened", "0")) == "1" for e in events):
                prioritized.append(cid)

        if prioritized:
            return prioritized[0]
        
        return candidates[0]

    def _build_context_summary(self, is_dnd: bool, is_group_muted: bool, is_domain_mismatch: bool, is_verified: bool, trust_score: float, evidence_ids: str) -> str:
        parts = []
        if is_dnd:
            parts.append("In User DND Quiet Window")
        if is_group_muted:
            parts.append("Group Muted By User")
        if is_domain_mismatch:
            parts.append("Domain Mismatch Detected")
        elif is_verified:
            parts.append("Verified Sender Account")
        
        parts.append(f"Trust Score: {round(trust_score, 2)}")
        if evidence_ids != "none":
            parts.append(f"Evidence Ref: {evidence_ids}")

        return " | ".join(parts)
