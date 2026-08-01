import os
import csv
from code.data_loader import DataLoader
from code.multimodal_processor import MultimodalProcessor
from code.context_trust_engine import ContextTrustEngine
from code.decision_router import DecisionRouter

def main():
    dataset_dir = "dataset"
    print("Initializing DataLoader...")
    loader = DataLoader(dataset_dir).load_all()
    
    print("Initializing MultimodalProcessor...")
    processor = MultimodalProcessor()
    
    print("Initializing ContextTrustEngine...")
    trust_engine = ContextTrustEngine(loader)
    
    print("Initializing DecisionRouter...")
    router = DecisionRouter()
    
    print(f"Routing {len(loader.messages)} messages from dataset...")
    output_rows = []
    
    for msg in loader.messages:
        ctx = loader.get_message_context(msg)
        multimodal_feat = processor.process_message(ctx)
        trust_eval = trust_engine.evaluate_context_and_trust(ctx, multimodal_feat)
        decision = router.route_message(ctx, multimodal_feat, trust_eval)
        
        output_rows.append({
            "message_id": decision["message_id"],
            "action": decision["action"],
            "message_type": decision["message_type"],
            "reason": decision["reason"],
            "confidence": decision["confidence"],
            "evidence_message_ids": decision["evidence_message_ids"]
        })
        
    output_path = os.path.join(dataset_dir, "output.csv")
    fieldnames = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
        
    print(f"Successfully generated {output_path} with {len(output_rows)} predictions.")

if __name__ == "__main__":
    main()
