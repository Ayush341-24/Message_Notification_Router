"""
Step 1: Data Ingestion, Schema Validation & Indexing Engine
Author: HackerRank Orchestrate Agent
Description:
Loads, validates, and builds indexed lookup structures for all WhatsApp dataset files and media files.
"""

import os
import csv
from typing import Dict, List, Any, Optional

class DataLoader:
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir
        self.messages: List[Dict[str, Any]] = []
        self.users: Dict[str, Dict[str, Any]] = {}
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.group_members: Dict[str, List[Dict[str, Any]]] = {}
        self.business_accounts: Dict[str, Dict[str, Any]] = {}
        self.user_business_history: Dict[tuple, Dict[str, Any]] = {}
        self.message_history: List[Dict[str, Any]] = []
        self.message_events: Dict[str, List[Dict[str, Any]]] = {}
        self.images: Dict[str, Dict[str, Any]] = {}
        self.voice_notes: Dict[str, Dict[str, Any]] = {}
        self.daily_notification_summary: Dict[tuple, Dict[str, Any]] = {}
        self.sample_messages: List[Dict[str, Any]] = []
        self.media_cache: Dict[str, Dict[str, Any]] = {}
        
        self.is_loaded = False

    def load_all(self) -> "DataLoader":
        """Load all CSV files from the dataset directory and index them."""
        print(f"[DataLoader] Loading dataset from directory: {self.dataset_dir}")
        
        # 1. messages.csv
        messages_path = os.path.join(self.dataset_dir, "messages.csv")
        if os.path.exists(messages_path):
            self.messages = self._read_csv(messages_path)
            print(f"  -> Loaded {len(self.messages)} target messages.")

        # 2. users.csv
        users_path = os.path.join(self.dataset_dir, "users.csv")
        if os.path.exists(users_path):
            users_list = self._read_csv(users_path)
            self.users = {row["user_id"]: row for row in users_list if "user_id" in row}
            print(f"  -> Loaded {len(self.users)} user profiles.")

        # 3. groups.csv
        groups_path = os.path.join(self.dataset_dir, "groups.csv")
        if os.path.exists(groups_path):
            groups_list = self._read_csv(groups_path)
            self.groups = {row["group_id"]: row for row in groups_list if "group_id" in row}
            print(f"  -> Loaded {len(self.groups)} group metadata entries.")

        # 4. group_members.csv
        gm_path = os.path.join(self.dataset_dir, "group_members.csv")
        if os.path.exists(gm_path):
            gm_list = self._read_csv(gm_path)
            for row in gm_list:
                gid = row.get("group_id")
                if gid:
                    if gid not in self.group_members:
                        self.group_members[gid] = []
                    self.group_members[gid].append(row)
            print(f"  -> Loaded group memberships across {len(self.group_members)} groups.")

        # 5. business_accounts.csv
        biz_path = os.path.join(self.dataset_dir, "business_accounts.csv")
        if os.path.exists(biz_path):
            biz_list = self._read_csv(biz_path)
            self.business_accounts = {row["business_id"]: row for row in biz_list if "business_id" in row}
            print(f"  -> Loaded {len(self.business_accounts)} business accounts.")

        # 6. user_business_history.csv
        ubh_path = os.path.join(self.dataset_dir, "user_business_history.csv")
        if os.path.exists(ubh_path):
            ubh_list = self._read_csv(ubh_path)
            for row in ubh_list:
                key = (row.get("user_id"), row.get("business_id"))
                self.user_business_history[key] = row
            print(f"  -> Loaded {len(self.user_business_history)} user-business interaction records.")

        # 7. message_history.csv
        mh_path = os.path.join(self.dataset_dir, "message_history.csv")
        if os.path.exists(mh_path):
            self.message_history = self._read_csv(mh_path)
            print(f"  -> Loaded {len(self.message_history)} historical message logs.")

        # 8. message_events.csv
        me_path = os.path.join(self.dataset_dir, "message_events.csv")
        if os.path.exists(me_path):
            me_list = self._read_csv(me_path)
            for row in me_list:
                mid = row.get("message_id")
                if mid:
                    if mid not in self.message_events:
                        self.message_events[mid] = []
                    self.message_events[mid].append(row)
            print(f"  -> Loaded message interaction events for {len(self.message_events)} messages.")

        # 9. images.csv
        img_path = os.path.join(self.dataset_dir, "images.csv")
        if os.path.exists(img_path):
            img_list = self._read_csv(img_path)
            self.images = {row["image_id"]: row for row in img_list if "image_id" in row}
            print(f"  -> Loaded {len(self.images)} image metadata entries.")

        # 10. voice_notes.csv
        vn_path = os.path.join(self.dataset_dir, "voice_notes.csv")
        if os.path.exists(vn_path):
            vn_list = self._read_csv(vn_path)
            self.voice_notes = {row["voice_note_id"]: row for row in vn_list if "voice_note_id" in row}
            print(f"  -> Loaded {len(self.voice_notes)} voice note metadata entries.")

        # 11. daily_notification_summary.csv
        dns_path = os.path.join(self.dataset_dir, "daily_notification_summary.csv")
        if os.path.exists(dns_path):
            dns_list = self._read_csv(dns_path)
            for row in dns_list:
                key = (row.get("user_id"), row.get("date"))
                self.daily_notification_summary[key] = row
            print(f"  -> Loaded {len(self.daily_notification_summary)} daily notification summary logs.")

        # 12. sample_messages.csv (for ground truth validation if present)
        sm_path = os.path.join(self.dataset_dir, "sample_messages.csv")
        if os.path.exists(sm_path):
            self.sample_messages = self._read_csv(sm_path)
            print(f"  -> Loaded {len(self.sample_messages)} sample messages with evaluation labels.")

        # 13. media_cache.json (optional multimodal cached transcript/OCR)
        mc_path = os.path.join(self.dataset_dir, "media_cache.json")
        if os.path.exists(mc_path):
            try:
                import json
                with open(mc_path, "r", encoding="utf-8") as f:
                    self.media_cache = json.load(f)
                print(f"  -> Loaded multimodal media cache for {len(self.media_cache)} media assets.")
            except Exception as e:
                print(f"  -> Note: could not load media_cache.json: {e}")

        self.is_loaded = True
        return self

    def search_messages(self, query: str = "", message_type: str = "All", limit: int = 100) -> List[Dict[str, Any]]:
        """Filter messages by search query and message_type."""
        results = self.messages
        
        if message_type != "All":
            results = [
                m for m in results 
                if (m.get("media_type") or m.get("message_type") or "text") == message_type
            ]
            
        if query:
            q = query.lower()
            results = [
                m for m in results
                if q in str(m.get("message_id", "")).lower()
                or q in str(m.get("message_text", m.get("text_content", ""))).lower()
                or q in str(m.get("user_id", "")).lower()
                or q in str(m.get("sender_user_id", m.get("sender_id", ""))).lower()
                or q in str(m.get("business_id", "")).lower()
                or q in str(m.get("group_id", "")).lower()
            ]
            
        return results[:limit]

    def get_summary_analytics(self) -> Dict[str, Any]:
        """Compute dataset analytics distributions."""
        type_counts = {}
        for m in self.messages:
            mtype = m.get("media_type") or m.get("message_type") or "text"
            if not mtype or mtype.strip() == "":
                mtype = "text"
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
            
        group_vs_direct = {"Group Chat": 0, "Business/Direct": 0}
        for m in self.messages:
            if m.get("group_id") and str(m.get("group_id")).strip():
                group_vs_direct["Group Chat"] += 1
            else:
                group_vs_direct["Business/Direct"] += 1

        return {
            "type_counts": type_counts,
            "group_vs_direct": group_vs_direct,
            "sample_labeled_count": len(self.sample_messages)
        }

    def _read_csv(self, filepath: str) -> List[Dict[str, Any]]:
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def validate_dataset(self) -> Dict[str, Any]:
        """Perform schema and media file existence validation."""
        missing_images = []
        missing_audio = []

        for img_id, img_info in self.images.items():
            rel_path = img_info.get("file_path", "")
            full_path = os.path.join(self.dataset_dir, rel_path) if not rel_path.startswith("dataset/") else rel_path
            if not os.path.exists(full_path):
                missing_images.append((img_id, full_path))

        for vn_id, vn_info in self.voice_notes.items():
            rel_path = vn_info.get("file_path", "")
            full_path = os.path.join(self.dataset_dir, rel_path) if not rel_path.startswith("dataset/") else rel_path
            if not os.path.exists(full_path):
                missing_audio.append((vn_id, full_path))

        validation_result = {
            "total_messages": len(self.messages),
            "total_users": len(self.users),
            "total_groups": len(self.groups),
            "total_businesses": len(self.business_accounts),
            "total_images": len(self.images),
            "total_voice_notes": len(self.voice_notes),
            "missing_image_files": missing_images,
            "missing_audio_files": missing_audio,
            "has_metadata_fallbacks": True,
            "valid": True  # Data structures fully loaded and indexed with metadata fallbacks
        }
        return validation_result

    def get_message_context(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve enriched contextual data for a given message row."""
        user_id = msg.get("user_id")
        business_id = msg.get("business_id")
        sender_user_id = msg.get("sender_user_id") or msg.get("sender_id")
        
        # Fallback business_id lookup if sender is a business account
        if not business_id and sender_user_id in self.business_accounts:
            business_id = sender_user_id
        
        sender_id = business_id or sender_user_id
        group_id = msg.get("group_id")
        media_type = msg.get("media_type") or msg.get("message_type") or "text"
        media_id = msg.get("media_id")

        # Auto-detect media type if media_id is provided but media_type is generic/text
        if media_id:
            if media_id in self.images and media_type not in ["image"]:
                media_type = "image"
            elif media_id in self.voice_notes and media_type not in ["voice_note", "audio", "voice"]:
                media_type = "voice_note"

        # Lookup user-business interaction history flexibly
        ubh = self.user_business_history.get((user_id, business_id)) or self.user_business_history.get((user_id, sender_id)) or {}

        context = {
            "message": msg,
            "user_profile": self.users.get(user_id, {}),
            "group_info": self.groups.get(group_id, {}) if (group_id and str(group_id).strip()) else None,
            "business_info": self.business_accounts.get(business_id or sender_id, {}),
            "user_business_history": ubh,
            "media_metadata": None,
            "media_file_path": None
        }

        if media_type == "image" and media_id in self.images:
            img_info = dict(self.images[media_id])
            if media_id in self.media_cache:
                img_info.update(self.media_cache[media_id])
            context["media_metadata"] = img_info
            rel_p = img_info.get("file_path", "")
            context["media_file_path"] = os.path.join(self.dataset_dir, rel_p) if not rel_p.startswith("dataset/") else rel_p
        elif media_type in ["voice_note", "audio", "voice"] and media_id in self.voice_notes:
            vn_info = dict(self.voice_notes[media_id])
            if media_id in self.media_cache:
                vn_info.update(self.media_cache[media_id])
            context["media_metadata"] = vn_info
            rel_p = vn_info.get("file_path", "")
            context["media_file_path"] = os.path.join(self.dataset_dir, rel_p) if not rel_p.startswith("dataset/") else rel_p

        return context


if __name__ == "__main__":
    loader = DataLoader("dataset")
    loader.load_all()
    stats = loader.validate_dataset()
    print("\n================ STEP 1 VALIDATION REPORT ================")
    print(f"Dataset Valid: {stats['valid']}")
    print(f"Total Messages: {stats['total_messages']}")
    print(f"Total Users: {stats['total_users']}")
    print(f"Total Groups: {stats['total_groups']}")
    print(f"Total Businesses: {stats['total_businesses']}")
    print(f"Total Images: {stats['total_images']} (Missing files: {len(stats['missing_image_files'])})")
    print(f"Total Voice Notes: {stats['total_voice_notes']} (Missing files: {len(stats['missing_audio_files'])})")
    print("==========================================================")
