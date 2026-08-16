import re

def clean_plate(text):
    if not text:
        return None
    cleaned = re.sub(r'[^A-Za-z0-9]', '', str(text))
    return cleaned if cleaned else None

def build_payload(video_id: int, final_report: dict, vehicle_images: dict):
    vehicles = []
    for key, data in final_report.items():
        track_id = data.get("track_id")
        
        vehicles.append({
            "track_id": track_id,
            "plate_number": clean_plate(data.get("plate_text")),
            "plate_confidence": round(float(data.get("plate_conf", 0.0)), 2),
            "color": str(data.get("color")).lower() if data.get("color") else None,
            "color_confidence": round(float(data.get("color_conf", 0.0)), 2),
            "type": str(data.get("type")).lower() if data.get("type") else None,
            "type_confidence": round(float(data.get("type_conf", 0.0)), 2),
            "model": str(data.get("make_model")) if data.get("make_model") else None,
            "make_model_confidence": round(float(data.get("make_model_conf", 0.0)), 2),
            "vehicle_image_path": vehicle_images.get(track_id),
        })

    return {
        "video_id": video_id,
        "processed_video_path": f"processed/{video_id}/processed.mp4",
        "vehicles": vehicles,
    }