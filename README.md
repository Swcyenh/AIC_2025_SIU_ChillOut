- Main source code:
    + Write as class:
        + Remember to commend input output, and the purpose of the class
    + OCR, OD, CD:
        + Each file is a video:
        + Json {
            "video_name": "video1.mp4",
            "frames": [
                {
                    "frame_number": 1,
                    "objects": [
                        {
                            "class": "car",
                            "bbox": [x_min, y_min, x_max, y_max],
                            "confidence": 0.95
                        },
                        ...
                    ]
                },
                ...
            ]
        }
- API
- Documentation:
    + Write a README.md file

Deadline: 0 week from now - Finished 
- Dataset:
    L01 in /dataset/AIC_2024, /dataset/AIC2024/original_dataset/0/videos/Videos_L01/video/
- New task:
    + Phuc: Implement Embedding model, Feature extraction -> Class
    + Leo: Implement OCR, OD, UI (after api done)
    + Trieu, Anh: Rewrite API (Flask -> FastAPI), test repsonse time
    + Swcyen: Pylunce, Qdrant, CD, Whisper2, DRES

- Tmux:
    + Create a tmux session for each person
    + Use the command `tmux new -s <session_name>` to create a new session
    + Use `tmux attach -t <session_name>` to attach to a session
    + Use `tmux ls` to list all sessions
    + Use `tmux kill-session -t <session_name>` to kill a session

- Task need to be done before tuesday:
    + Extract: Eva, Long, DFN5B, SigLIP2 on full dataset
    + API: Update image and feature path
    + OCR, OD, ASR, CD: Rerun on full batch 0 dataset
    + UI: add selection for SigLIP2