from flask import Flask, request, render_template, send_from_directory
import math
import requests
import time
import json

app = Flask(__name__)
app.config['APPLICATION_ROOT'] = '/siu_chillout_2'

### Load FPS results
with open("/workspace/competitions/AIC_2025/SIU_ChillOut/Swcyen/Playground/video_fps_0.json", "r") as f:
    fps_results = json.load(f)

### Image Download Route
@app.route('/img/<path:filename>')
def download_file(filename):
    directory = "/".join(filename.split("/")[:-2]) + "/" + filename.split("/")[-2]
    video_name = filename.split("/")[-1]
    if directory[-1] == ".":
        return send_from_directory(directory="/" + directory[:-1], path=video_name)
    return send_from_directory(directory="/" + directory, path=video_name)

### Video Download Route
@app.route('/video/<path:filename>')
def download_video(filename):
    directory = "/".join(filename.split("/")[:-1])
    video_name = filename.split("/")[-1]
    return send_from_directory(directory="/" + directory, path=video_name)

### Index Route
@app.route('/', methods=['GET', 'POST'])
def index():
    files = []
    return render_template('index.html', files=files)

### Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model = request.form.get('model', '')
        text = request.form.get('text', '')
        trans = request.form.get('trans', 0)
        temporal = request.form.get('temporal', 0)
        del_keyframes = request.form.get('delKeyframes', 0)
        del_keyframes = 1 if del_keyframes == '1' else 0 # Nếu tick thì 1, không tick thì 0

        if trans != 0:
            output = f"https://api.siu.edu.vn/siu_chillout_8/translate?text={text}"
            text = requests.get(output).json()["translated_text"]
                
        if temporal != 0:
            output = f"https://api.siu.edu.vn/siu_chillout_7/search_temporal?text={text}&model={model}"
            result = requests.get(output).json()
            result = result.get('results', result) 
            lst_video_name = [(res['video_name'], res['keyframe_id']) for res in result if 'video_name' in res and 'keyframe_id' in res]

            files = []
            for video in lst_video_name:
                folder_name = video[0][:-4]
                if "K" in folder_name:
                    key_frame_path = f"/dataset/AIC_2025/SIU_Pumpking/1/frames/autoshot/Keyframes_{folder_name[:3]}/keyframes/{folder_name}/{video[1]}.jpg"
                else:
                    key_frame_path = f"/dataset/AIC_2025/SIU_Pumpking/0/frames/autoshot/Keyframes_{folder_name[:3]}/keyframes/{folder_name}/{video[1]}.jpg"
                fps = fps_results.get(f"{folder_name}.mp4", 25) 
                true_key_frame = int(video[1])
                keyframe_time = true_key_frame / fps
                mi = str(int(keyframe_time) // 60).zfill(2)
                se = str(int(keyframe_time) % 60).zfill(2)
                right_id = f"{mi}:{se},Videos_{folder_name[:3]},{folder_name},{video[1]}"
                files.append((key_frame_path, right_id, int(keyframe_time)))
            return render_template('index.html', files=files)
        elif model == "Combine":
            output = f"https://api.siu.edu.vn/siu_chillout_7/search_rrf?text={text}"
            result = requests.get(output).json()
            result = result.get('results', result) 
            lst_video_name = [(res['video_name'], res['keyframe_id']) for res in result if 'video_name' in res and 'keyframe_id' in res]
            print(f"Extracted video names and keyframe IDs: {lst_video_name}")
            files = []
            for video in lst_video_name:
                folder_name = video[0][:-4]
                if "K" in folder_name:
                    key_frame_path = f"/dataset/AIC_2025/SIU_Pumpking/1/frames/autoshot/Keyframes_{folder_name[:3]}/keyframes/{folder_name}/{video[1]}.jpg"
                else:
                    key_frame_path = f"/dataset/AIC_2025/SIU_Pumpking/0/frames/autoshot/Keyframes_{folder_name[:3]}/keyframes/{folder_name}/{video[1]}.jpg"
                fps = fps_results.get(f"{folder_name}.mp4", 25)  
                true_key_frame = int(video[1])
                keyframe_time = true_key_frame / fps
                mi = str(int(keyframe_time) // 60).zfill(2)
                se = str(int(keyframe_time) % 60).zfill(2)
                right_id = f"{mi}:{se},Videos_{folder_name[:3]},{folder_name},{video[1]}"
                files.append((key_frame_path, right_id, int(keyframe_time)))
            return render_template('index.html', files=files)
        else:
            if model == "MetaCLIP":
                output = f"https://api.siu.edu.vn/siu_chillout_4/predict?text={text}"
            elif model == "DFN5B":
                output = f"https://api.siu.edu.vn/siu_chillout_5/predict?text={text}"
            elif model == "EvaCLIP":
                output = f"https://api.siu.edu.vn/siu_chillout_6/predict?text={text}"
            else:
                return render_template('index.html', files=[], error="Please select a model.")

        try:
            result = requests.get(output).json()
            result = result.get('results', result) 
        except Exception as e:
            print(f"Error calling API: {e}")
            return render_template('index.html', files=[], error="Error calling API.")
        
        # --- Lọc Keyframes_L26 nếu checkbox tick ---
        if del_keyframes != 0:
            result = [res for res in result if not res.get('video_name',"").upper().startswith("L26")]

        try:
            lst_video_name = [(res['video_name'], res['keyframe_id']) for res in result if 'video_name' in res and 'keyframe_id' in res]
        except Exception as e:
            print(f"Error processing the results: {e}")
            return render_template('index.html', files=[], error="Error processing the results.")

        files = []
        files = []
        for video in lst_video_name:
            folder_name = video[0][:-4]
            if "K" in folder_name:
                key_frame_path = f"/dataset/AIC_2025/SIU_Pumpking/1/frames/autoshot/Keyframes_{folder_name[:3]}/keyframes/{folder_name}/{video[1]}.jpg"
            else:
                key_frame_path = f"/dataset/AIC_2025/SIU_Pumpking/0/frames/autoshot/Keyframes_{folder_name[:3]}/keyframes/{folder_name}/{video[1]}.jpg"
            fps = fps_results.get(f"{folder_name}.mp4", 25) 
            true_key_frame = int(video[1])
            keyframe_time = true_key_frame / fps
            mi = str(int(keyframe_time) // 60).zfill(2)
            se = str(int(keyframe_time) % 60).zfill(2)
            right_id = f"{mi}:{se},Videos_{folder_name[:3]},{folder_name},{video[1]}"
            files.append((key_frame_path, right_id, int(keyframe_time)))
        return render_template('index.html', files=files)
    else:
        return render_template('index.html', files=[])

if __name__ == "__main__":
    app.run("0.0.0.0", port=8600, debug=True)
