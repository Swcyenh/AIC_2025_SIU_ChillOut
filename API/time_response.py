import json
import os
from datetime import datetime, timedelta

def update_response_log(file_path, model, text, response_time, date_time):
    """Cập nhật file log response time (chạy nền, không block request)."""
    try:
        # Parse thời gian hiện tại từ tham số
        now = datetime.strptime(date_time, "%Y-%m-%dT%H:%M:%S") if "T" in date_time else datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")

        # Bản ghi mới
        record = {
            "date_time": date_time,
            "model" : model,
            "text": text,
            "response_time": response_time
        }

        # Nếu file tồn tại thì đọc, không thì tạo mới
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = {"model": model, "records": []}

        # Lọc lại chỉ giữ bản ghi trong vòng 7 ngày gần nhất
        seven_days_ago = now - timedelta(days=7)
        filtered_records = []

        for r in data.get("records", []):
            try:
                r_time = datetime.strptime(r["date_time"], "%Y-%m-%dT%H:%M:%S") if "T" in r["date_time"] else datetime.strptime(r["date_time"], "%Y-%m-%d %H:%M:%S")
                if r_time >= seven_days_ago:
                    filtered_records.append(r)
            except:
                pass  # Bỏ qua nếu sai định dạng

        # Gán lại danh sách records đã lọc
        data["records"] = filtered_records

        # Thêm bản ghi mới
        data["records"].append(record)

        # Ghi lại file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"[WARNING] Logging failed: {e}")

