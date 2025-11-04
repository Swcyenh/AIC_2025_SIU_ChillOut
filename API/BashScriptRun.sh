#!/bin/bash

ROOT_DIR="/workspace/competitions/AIC_2025/SIU_ChillOut/Main_Source"

usage() {
  echo "Usage: $0 status|start|stop|rerun [model1 model2 ...]"
  echo "  If no models specified, applies to all"
  exit 1
}

ACTION=$1
shift

# Ép ACTION về chữ thường
ACTION=$(echo "$ACTION" | tr '[:upper:]' '[:lower:]')

if [[ "$ACTION" != "status" && "$ACTION" != "start" && "$ACTION" != "stop" && "$ACTION" != "rerun" ]]; then
  usage
fi

MODELS=("$@")

# ==== CHECK PORT FUNCTION ====
check_port() {
  local port=$1
  if ss -lnt | grep -q ":$port "; then
    echo "✅ Port $port is OPEN"
  else
    echo "❌ Port $port is CLOSED"
  fi
}

# ==== START FUNCTION ====
# 8594
start_metaclip() {
  echo "Starting MetaCLIP API"
  cd $ROOT_DIR/API
  CUDA_VISIBLE_DEVICES=1 taskset -c 16-30 gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8594 MetaClip_FastAPI:app > $ROOT_DIR/API/API_Run_Logs/metaclip.log 2>&1 &
}

# 8595
start_jinaclip() {
  echo "Starting JinaCLIP API"
  cd $ROOT_DIR/API
  CUDA_VISIBLE_DEVICES=1 taskset -c 16-30 gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8595 JINA_CLIP:app > $ROOT_DIR/API/API_Run_Logs/jinaclip.log 2>&1 &
}

# 8596
start_siglip2() {
  echo "Starting SigLIP2 API"
  cd $ROOT_DIR/API
  CUDA_VISIBLE_DEVICES=2 taskset -c 16-30 gunicorn -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8596 SigLIP_2:app > $ROOT_DIR/API/API_Run_Logs/siglip2.log 2>&1 &
}

# 8597
start_combine() {
  echo "Starting Combine API"
  cd $ROOT_DIR/API
  CUDA_VISIBLE_DEVICES=2 taskset -c 16-30 gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8597 Combine_FastAPI_2:app > $ROOT_DIR/API/API_Run_Logs/combine.log 2>&1 &
}

# 8598
start_translate() {
  echo "Starting Translate API"
  cd $ROOT_DIR/API
  CUDA_VISIBLE_DEVICES=1 taskset -c 16-30 gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8598 Translate_FastAPI:app > $ROOT_DIR/API/API_Run_Logs/translate.log 2>&1 &
}

# 8599

# 8600
start_web() {
  echo "Starting Web"
  cd $ROOT_DIR/Web
  CUDA_VISIBLE_DEVICES=0 taskset -c 16-30 gunicorn -w 4 -b 0.0.0.0:8600 server:app > $ROOT_DIR/API/API_Run_Logs/Web.log 2>&1 &
}

# ==== STOP FUNCTION ====
# 8594
stop_metaclip() {
  echo "Stopping MetaCLIP API"
  fuser -k 8594/tcp
}

# 8595
stop_jinaclip() {
  echo "Stopping JinaCLIP API"
  fuser -k 8595/tcp
}

# 8596
stop_siglip2() {
  echo "Stopping SigLIP2 API"
  fuser -k 8596/tcp
}

# 8597
stop_combine() {
  echo "Stopping Combine API"
  fuser -k 8597/tcp
}

# 8598
stop_translate() {
  echo "Stopping Translate API"
  fuser -k 8598/tcp
}

# 8599

# 8600
stop_web() {
  echo "Stopping Web"
  fuser -k 8600/tcp
}

mkdir -p $ROOT_DIR/API/API_Run_Logs

if [[ "$ACTION" = "start" || "$ACTION" = "rerun" ]]; then
  # Load micromamba/conda init
  eval "$(micromamba shell hook -s bash)"

  # Kiểm tra đã activate conda chưa
  if [ "$CONDA_DEFAULT_ENV" != "lavis" ]; then
    echo "🔹 lavis is not active, activating 'lavis'..."
    micromamba activate lavis
  else
    echo "✅ Conda environment already active: $CONDA_DEFAULT_ENV"
  fi
fi

# ==== RUN ACTION ====
run_action() {
  local model=$1
  case "${model,,}" in
    meta*clip) port=8594; fn="metaclip" ;;
    jina*clip) port=8595; fn="jinaclip" ;;
    siglip2) port=8596; fn="siglip2" ;;
    translate) port=8598; fn="translate" ;;
    combine) port=8597; fn="combine" ;;
    web) port=8600; fn="web" ;;
    *) echo "Unknown model: $model"; return ;;
  esac

  if [ "$ACTION" = "status" ]; then
    check_port $port
  elif [ "$ACTION" = "start" ]; then
    start_${fn}
  elif [ "$ACTION" = "stop" ]; then
    stop_${fn}
  elif [ "$ACTION" = "rerun" ]; then
    echo "🔄 Rerunning $model..."
    stop_${fn}
    sleep 2
    start_${fn}
  fi
}

# ==== MAIN LOOP ====
if [ ${#MODELS[@]} -eq 0 ]; then
  # Chạy toàn bộ 
  run_action MetaCLIP
  run_action JinaCLIP
  run_action SigLIP2
  #run_action Combine
  run_action Translate
  run_action Web
elif [ "${MODELS[0]}" = "0" ]; then
  # Bộ được chọn sẵn khi truyền vào "0"
  run_action MetaCLIP
  run_action Translate
  run_action Web
elif [ "${MODELS[0]}" = "1" ]; then
  # Bộ được chọn sẵn khi truyền vào "1"
  run_action MetaCLIP
  run_action JinaCLIP
  run_action SigLIP2
  run_action Translate
  run_action Web
elif [ "${MODELS[0]}" = "2" ]; then
  # Bộ được chọn sẵn khi truyền vào "2"
  run_action JinaCLIP
  run_action SigLIP2
else
  # Chạy mô hình theo thứ tự truyền vào
  for m in "${MODELS[@]}"; do
    run_action "$m"
  done
fi

echo "Done."
