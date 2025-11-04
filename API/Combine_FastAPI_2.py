import logging
import time
import collections
import math
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Query, HTTPException, Request, requests
from starlette.responses import Response, StreamingResponse
import os
from datetime import datetime
import threading
import requests 

# Import hàm ghi log thời gian
from time_response_combine import update_response_log

# === Logging setup ===
LOG_FOLDER = "/workspace/competitions/AIC_2025/SIU_ChillOut/Main_Source/API/API_Run_Logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_FOLDER}/combine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === FastAPI init ===
app = FastAPI(title="Combine")

# === Data structure ===
class SearchResults:
    def __init__(self, keyframe_id, score, video_name):
        self.keyframe_id = int(keyframe_id)
        self.score = float(score)
        self.video_name = video_name

    def to_dict(self):
        return {
            "keyframe_id": self.keyframe_id,
            "score": round(self.score, 5),
            "video_name": self.video_name
        }

# === Combine / RRF functions ===
def get_video_hash_map(results, time_quantizer):
    shmap = collections.defaultdict(dict)
    for result in results:
        quantized_time = result.keyframe_id // time_quantizer
        shmap[result.video_name][quantized_time] = result
    return shmap

def combine_results_temporal(results_list, top_k, time_quantizer, time_interval, min_num_lists=1):
    n_lists_to_merge = len(results_list)
    qi = round(time_interval / time_quantizer)
    merged_results = []
    max_scores = [max((r.score for r in results), default=1) for results in results_list]
    shmap_list = [get_video_hash_map(results, time_quantizer) for results in results_list]
    video_id_counts = collections.Counter()
    for shmap in shmap_list:
        video_id_counts.update(shmap.keys())
    video_ids = {vid for vid, count in video_id_counts.items() if count >= min_num_lists}

    for video_name in video_ids:
        hm_list = [shmap.get(video_name, {}) for shmap in shmap_list]
        quantized_times = set()
        for hm in hm_list:
            quantized_times.update(hm.keys())
        for qtime in quantized_times:
            best_results = []
            for i in range(n_lists_to_merge):
                matching_result = None
                for offset in range(-qi, qi + 1):
                    matching_result = hm_list[i].get(qtime + offset)
                    if matching_result:
                        break
                best_results.append(matching_result)
            scores = [r.score / max_scores[i] for i, r in enumerate(best_results) if r]
            if scores:
                combined_score = math.prod(scores)
                if combined_score > 0:
                    keyframe_id = next((r.keyframe_id for r in best_results if r), None)
                    merged_results.append(SearchResults(keyframe_id, combined_score, video_name))
    merged_results.sort(key=lambda x: x.score, reverse=True)
    return merged_results[:top_k]

def convert_to_search_results(json_response):
    return [SearchResults(d['keyframe_id'], d['score'], d['video_name']) for d in json_response]

def merge_results_rrf_with_boost(results_list, top_k, k_rrf=100, n=3):
    fused_scores = collections.defaultdict(lambda: {'score': 0.0, 'video_name': None})
    n_hits = len(results_list)

    def process_results(rs):
        max_rank = min(top_k, len(rs))
        for rank, result in enumerate(rs[:max_rank]):
            keyframe_id = int(result['keyframe_id'])
            score = float(result['score'])
            video_name = result['video_name']
            rrf_score = k_rrf / (rank + k_rrf)
            if rank < n:
                rrf_score *= n_hits
            fused_scores[keyframe_id]['score'] += rrf_score
            if fused_scores[keyframe_id]['video_name'] is None:
                fused_scores[keyframe_id]['video_name'] = video_name

    for rs in results_list:
        process_results(rs)

    sorted_results = sorted(fused_scores.items(), key=lambda x: -x[1]['score'])
    return [SearchResults(k, v['score'], v['video_name']) for k, v in sorted_results[:top_k]]

# === Model API endpoints ===
model_ports = {
    "MetaCLIP": "https://api.siu.edu.vn/siu_chillout_4/predict",
    "DFN5B": "https://api.siu.edu.vn/siu_chillout_5/predict",
    "EvaCLIP": "https://api.siu.edu.vn/siu_chillout_6/predict",
}

# === Middleware log thời gian + ghi log nền ===
@app.middleware("http")
async def log_request_response(request: Request, call_next):
    start_time = time.time()
    response: Response = await call_next(request)
    duration = time.time() - start_time

    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    text_param = request.query_params.get("text", "") if request.method == "GET" else ""

    req_type = None
    if request.url.path.startswith("/search_temporal"):
        req_type = "temporal"
    elif request.url.path.startswith("/search_rrf"):
        req_type = "rrf"

    # Log thời gian xử lý
    if req_type:
        logger.info(f"{request.method} {request.url.path} took {duration:.4f}s | text: {text_param}")

    # Trả response ngay
    resp_body = b"".join([chunk async for chunk in response.body_iterator])
    fast_response = StreamingResponse(
        iter([resp_body]),
        status_code=response.status_code,
        headers=dict(response.headers)
    )

    # Ghi log JSON nền
    if req_type:
        time_json_file = "/workspace/competitions/AIC_2025/SIU_ChillOut/Main_Source/API/API_Response_Logs/Combine_response_time.json"

        threading.Thread(
            target=update_response_log,
            args=(time_json_file, "Combine", text_param, duration, date_time_str, req_type),
            daemon=True
        ).start()

    return fast_response

# === Search endpoints ===
@app.get("/search_temporal")
async def search_temporal(text: str = Query(...), model: str = Query(...)):
    text_parts = [i.strip() for i in text.split('.') if i.strip()]
    if not text_parts:
        return JSONResponse(status_code=400, content={"error": "No valid text parts"})
    if model not in model_ports:
        return JSONResponse(status_code=400, content={"error": "Invalid model"})

    temporal_queries = []
    for part in text_parts:
        url = f"{model_ports[model]}?text={part}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", []) if isinstance(data, dict) else data
            if results: temporal_queries.append(convert_to_search_results(results))
    if not temporal_queries:
        return JSONResponse(status_code=404, content={"error": "No results"})

    temporal_ans = combine_results_temporal(
        temporal_queries, top_k=200, time_quantizer=200.0, time_interval=300.0
    )
    return JSONResponse(content={
        "success": True,
        "request_type": "temporal",
        "results": [r.to_dict() for r in temporal_ans]
    })

@app.get("/search_rrf")
async def search_rrf(text: str = Query(...)):
    results = []
    for model, port in model_ports.items():
        response = requests.get(f"{port}?text={text}")
        if response.status_code == 200:
            data = response.json()
            results.append(data.get("results", data) if isinstance(data, dict) else data)
    if not results:
        return JSONResponse(status_code=404, content={"error": "No results from models"})

    merged_results = merge_results_rrf_with_boost(results, top_k=200)
    return JSONResponse(content={
        "success": True,
        "request_type": "rrf",
        "results": [r.to_dict() for r in merged_results]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Combine_FastAPI_optimized:app", host="0.0.0.0", port=8597, reload=False)
