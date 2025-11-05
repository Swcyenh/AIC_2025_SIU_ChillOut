import sys
sys.path.append('/workspace/competitions/AIC_2025/SIU_ChillOut/phuc/Models')
sys.path.append('/workspace/competitions/AIC_2025/SIU_ChillOut/Main_Source')

import os
import uvicorn
import numpy as np
from tqdm import tqdm
import faiss
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response, StreamingResponse
import time
import pickle
import logging
import threading
from datetime import datetime
from Class.DFN5B import DFN5B
from Utils.Class.Utils import Utils
from fastapi import UploadFile, File
from API.time_response import update_response_log

# === Logging setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === FAISS indexing function (giống MetaCLIP) ===
def indexing_methods_faiss(
    model_features_path: str,
    size_of_embedding: int = None,
    subset_folders: list = None,
    index_type: str = "FlatIP",
    hnsw_m: int = 32,
    index_save_path: str = None,
    N_THREADS: int = 20
):
    faiss.omp_set_num_threads(N_THREADS)
    
    if index_save_path and os.path.exists(index_save_path):
        logger.info("Loading FAISS index from disk...")
        faiss_db = faiss.read_index(index_save_path, faiss.IO_FLAG_MMAP)
        with open(index_save_path + '_db.pkl', 'rb') as f:
            db = pickle.load(f)
        return db, faiss_db

    logger.info("Building FAISS index...")
    
    # Detect embedding size from first .npy file if not provided
    paths = [os.path.join(model_features_path, f) for f in os.listdir(model_features_path) if f.endswith('.npy')]
    if not paths:
        raise ValueError(f"No .npy files found in {model_features_path}")

    if size_of_embedding is None:
        feats_arr = np.load(paths[0])
        if feats_arr.ndim == 1:
            feats_arr = feats_arr.reshape(1, -1)
        elif feats_arr.ndim != 2:
            raise ValueError(f"Invalid shape for features in file {paths[0]}: {feats_arr.shape}")
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
    for feat_npy in tqdm(paths, desc='Adding features to index'):
        video_name = os.path.splitext(os.path.basename(feat_npy))[0]
        if subset_folders and not any(folder in video_name for folder in subset_folders):
            continue
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
        logger.info("Saving index to disk...")
        faiss.write_index(faiss_db, index_save_path)
        with open(index_save_path + '_db.pkl', 'wb') as f:
            pickle.dump(db, f)

    return db, faiss_db

# === Initialize FastAPI app ===
app = FastAPI(title="DFN5B Search API", root_path='/siu_chillout_4')

# === CORS middleware ===
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
    try:
        text_param = await request.json()
        text_param = text_param.get("text", "")
    except:
        text_param = request.query_params.get("text", "")

    # Log thời gian xử lý cho /predict
    if request.url.path.startswith("/predict"):
        logger.info(f"{request.method} {request.url.path} took {duration:.4f}s | text: {text_param}")

        # Ghi log nền
        time_json_file = "/workspace/competitions/AIC_2025/SIU_ChillOut/HighChill/API_Response_Logs/DFN5B_response_time.json"
        threading.Thread(
            target=update_response_log,
            args=(time_json_file, "DFN5B", text_param, duration, date_time_str),
            daemon=True
        ).start()

    resp_body = b"".join([chunk async for chunk in response.body_iterator])
    return StreamingResponse(iter([resp_body]), status_code=response.status_code, headers=dict(response.headers))

# === Request model ===
class SearchRequest(BaseModel):
    text: str

# === Load FAISS & DFN5B model ===
FEATURES_PATH = "/dataset/AIC_2025/SIU_ChillOut/Feature/DFN5BS"
KEYFRAME_FOLDER_PATHS = [
    "/dataset/AIC_2025/SIU_Pumpking/0/frames/autoshot",
    "/dataset/AIC_2025/SIU_Pumpking/1/frames/autoshot"
]
db, faiss_db = indexing_methods_faiss(FEATURES_PATH, 1024, subset_folders=None, index_type="FlatIP")
dfn5b = DFN5B()
utils = Utils(dfn5b, faiss_db=faiss_db, db=db, keyframe_folder_paths=KEYFRAME_FOLDER_PATHS)

# === API predict POST & GET ===
@app.post("/predict")
async def predict_post(request: SearchRequest):
    text = request.text
    logger.info(f"Received POST query: {text}")
    try:
        if "/L" in text:
            results = utils.search_image(text)
        else:
            results = utils.search_text(text)
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    return {"success": True, "request_type": "DFN5B", "results": results}

@app.get("/predict")
async def predict_get(text: str):
    logger.info(f"Received GET query: {text}")
    try:
        if "/L" in text:
            results = utils.search_image(text)
        else:
            results = utils.search_text(text)
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    return {"success": True, "request_type": "DFN5B", "results": results}

# === Run ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8595)
