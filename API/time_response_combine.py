import json
import os
import sys
from datetime import datetime, timedelta

def update_response_log(file_path, model, text, response_time, date_time_str, request_type):
    """
    Cập nhật response_time.json cho Combine.
    file_path: path file JSON
    model: 'Combine'
    text: query
    response_time: thời gian xử lý
    date_time_str: ISO format 'YYYY-MM-DDTHH:MM:SS'
    request_type: 'temporal' hoặc 'rrf'
    """
    # Parse thời gian
    date_time_obj = datetime.strptime(date_time_str, "%Y-%m-%dT%H:%M:%S")

    # Bản ghi mới
    record = {
        "date_time": date_time_str,
        "model": model,
        "request_type": request_type,
        "text": text,
        "response_time": response_time
    }

    # Nếu file tồn tại thì đọc, không thì tạo mới
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {"model": "Combine", "records": []}
    else:
        data = {"model": "Combine", "records": []}

    # Lọc lại records: chỉ giữ các bản ghi trong vòng 7 ngày
    seven_days_ago = date_time_obj - timedelta(days=7)
    filtered_records = []
    for r in data.get("records", []):
        try:
            r_time = datetime.strptime(r["date_time"], "%Y-%m-%dT%H:%M:%S")
            if r_time >= seven_days_ago:
                filtered_records.append(r)
        except Exception:
            continue

    # Thêm bản ghi mới
    filtered_records.append(record)
    data["records"] = filtered_records

    # Ghi lại file JSON
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Nếu gọi trực tiếp từ command line
if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python time_response_combine.py <file_path> <model> <text> <response_time> <date_time> <request_type>")
        sys.exit(1)
    _, file_path, model, text, response_time, date_time_str, request_type = sys.argv
    update_response_log(file_path, model, text, float(response_time), date_time_str, request_type)
