import sys
sys.path.append('/workspace/competitions/AIC_2025/SIU_ChillOut/phuc/Models')
sys.path.append('/workspace/competitions/AIC_2025/SIU_ChillOut/Main_Source')

import os
import time
import pickle
import logging
import threading
import numpy as np
from tqdm import tqdm
from datetime import datetime
import faiss
from pydantic import BaseModel
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response, StreamingResponse
from siglip2_encoder import SigLIP2Encoder
from Utils.Class.Utils import Utils
from API.time_response import update_response_log
import uuid
import requests

# === Logging setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# === FAISS indexing ===
def indexing_methods_faiss(
    model_features_path, 
    subset_folders=None,
    index_type="FlatIP", 
    hnsw_m=32,
    index_save_path=None, 
    N_THREADS=20):
    faiss.omp_set_num_threads(N_THREADS)

    if index_save_path and os.path.exists(index_save_path):
        logger.info("Loading FAISS index from disk...")
        faiss_db = faiss.read_index(index_save_path, faiss.IO_FLAG_MMAP)
        with open(index_save_path + '_db.pkl', 'rb') as f:
            db = pickle.load(f)
        return db, faiss_db

    logger.info("Building FAISS index...")

    file_list = sorted([f for f in os.listdir(model_features_path) if f.endswith('.npy')])
    if not file_list:
        raise ValueError(f"No .npy files found in {model_features_path}")

    # xác định embedding size
    first_feat_file = os.path.join(model_features_path, file_list[0])
    feats_arr = np.load(first_feat_file)
    if feats_arr.ndim == 1:
        feats_arr = feats_arr.reshape(1, -1)
    elif feats_arr.ndim == 3 and feats_arr.shape[1] == 1:
        feats_arr = feats_arr.squeeze(axis=1)
    elif feats_arr.ndim != 2:
        raise ValueError(f"Invalid shape for {first_feat_file}: {feats_arr.shape}")

    size_of_embedding = feats_arr.shape[1]
    logger.info(f"Detected embedding size: {size_of_embedding}")

    if index_type == "HNSW":
        faiss_db = faiss.IndexHNSWFlat(size_of_embedding, hnsw_m)
    elif index_type == "FlatIP":
        faiss_db = faiss.IndexFlatIP(size_of_embedding)
    elif index_type == "L2":
        faiss_db = faiss.IndexFlatL2(size_of_embedding)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    db = []
    paths = [os.path.join(model_features_path, f) for f in os.listdir(model_features_path) if f.endswith('.npy')]
    if not paths:
        raise ValueError(f"No .npy files found in {model_features_path}")
    
    for feat_npy in tqdm(paths, desc='Adding features to index'):
        video_name = os.path.splitext(os.path.basename(feat_npy))[0]
        if subset_folders and not any(folder in video_name for folder in subset_folders):
            continue
        print(video_name)
        feats_arr = np.load(feat_npy)
        if feats_arr.ndim == 1:
            feats_arr = feats_arr.reshape(1, -1)
        elif feats_arr.ndim != 2:
            raise ValueError(f"Invalid shape for features in file {feat_npy}: {feats_arr.shape}")
        for idx in range(feats_arr.shape[0]):
            db.append((video_name, idx))
        faiss_db.add(feats_arr.astype('float32'))
    db = dict(enumerate(db))

    if index_save_path:
        logger.info("Saving FAISS index...")
        faiss.write_index(faiss_db, index_save_path)
        with open(index_save_path + '_db.pkl', 'wb') as f:
            pickle.dump(db, f)

    return db, faiss_db


# === FastAPI App ===
app = FastAPI(title="SigLIP2 Search API", root_path="/siu_chillout_siglip2")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Middleware log thời gian + ghi log nền ===
@app.middleware("http")
async def log_request_response(request: Request, call_next):
    start_time = time.time()
    response: Response = await call_next(request)
    duration = time.time() - start_time

    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    text_param = request.query_params.get("text", "") if request.method == "GET" else ""

    if request.url.path.startswith("/predict"):
        logger.info(f"{request.method} {request.url.path} took {duration:.4f}s | text: {text_param}")

    resp_body = b"".join([chunk async for chunk in response.body_iterator])
    fast_response = StreamingResponse(iter([resp_body]),
                                      status_code=response.status_code,
                                      headers=dict(response.headers))

    if request.url.path.startswith("/predict"):
        time_json_file = "/workspace/competitions/AIC_2025/SIU_ChillOut/Main_Source/API/API_Response_Logs/SigLIP2_response_time.json"

        threading.Thread(
            target=update_response_log,
            args=(time_json_file, "SigLIP2", text_param, duration, date_time_str),
            daemon=True
        ).start()

    return fast_response


# === Load FAISS & Model ===
CLIP_FEATURES_PATH = "/dataset/AIC_2025/SIU_ChillOut/Feature/SIGLIP2"
KEYFRAME_FOLDER_PATHS = [
    "/dataset/AIC_2025/SIU_Pumpking/0/frames/autoshot",
    "/dataset/AIC_2025/SIU_Pumpking/1/frames/autoshot"
]

db, faiss_db = indexing_methods_faiss(CLIP_FEATURES_PATH, index_type="FlatIP")
sigLIP2 = SigLIP2Encoder()
utils = Utils(sigLIP2, faiss_db=faiss_db, db=db, keyframe_folder_paths=KEYFRAME_FOLDER_PATHS)


# === API ===
class SearchRequest(BaseModel):
    text: str
    del_keyframes: bool = False
    
class RelatedRequest(BaseModel):
    video_name: str
    frame_id: int
    topk: int = 200

SAVE_DIR = "/workspace/competitions/AIC_2025/SIU_ChillOut/Swcyen/Playground/img_from_img_search"
os.makedirs(SAVE_DIR, exist_ok=True)

# === API predict ===
@app.post("/predict")
async def predict(request: SearchRequest):
    text = request.text
    del_keyframes = request.del_keyframes
    logger.info(f"Received POST query: {text}")

    try:
        # Case 1: API link containing /dataset
        if "siu_chillout_10/img" in text and "/dataset" in text:
            text = text[text.index("/dataset"):]
            
        # Case 2: External image URL (download and save locally)
        elif text.lower().startswith("http") and text.lower().endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")):
            response = requests.get(text, timeout=10)
            response.raise_for_status()
            ext = os.path.splitext(text)[-1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                ext = ".jpg"
            local_path = os.path.join(SAVE_DIR, f"{uuid.uuid4().hex}{ext}")
            with open(local_path, "wb") as f:
                f.write(response.content)
            text = local_path  # replace query with local path

        # Decide search type
        if text.endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")) or "/dataset" in text:
            search_results = utils.search_image(text, del_keyframes=del_keyframes)
        else:
            search_results = utils.search_text(text, del_keyframes=del_keyframes)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    return {"success": True, "request_type": "MetaCLIP", "results": search_results}


@app.get("/predict")
async def predict_get(text: str):

    try:
        if "siu_chillout_10/img" in text and "/dataset" in text:
            text = text[text.index("/dataset"):]

        elif text.lower().startswith("http") and text.lower().endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")):

            response = requests.get(text, timeout=10)
            response.raise_for_status()
            ext = os.path.splitext(text)[-1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                ext = ".jpg"
            local_path = os.path.join(SAVE_DIR, f"{uuid.uuid4().hex}{ext}")
            with open(local_path, "wb") as f:
                f.write(response.content)
            text = local_path

        if text.endswith((".jpg", ".png", ".jpeg", ".gif", ".webp")) or "/dataset" in text:
            search_results = utils.search_image(text)
        else:
            search_results = utils.search_text(text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    return {"success": True, "request_type": "MetaCLIP", "results": search_results}


# === Run App ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8596)
