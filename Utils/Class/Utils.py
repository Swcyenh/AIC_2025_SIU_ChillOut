import os
import faiss
import tqdm
import numpy as np

class Utils:
    """
    Class to do utils shit
    Parameters:
        - model : object, model object
        - faiss_db : object, faiss db object
        - db : object, db object
        - keyframe_folder_path : str, path to keyframe folder
    """

    def __init__(self, model, faiss_db, db, keyframe_folder_paths):
        self.model = model
        self.faiss_db = faiss_db
        self.db = db
        self.keyframe_folder_paths = keyframe_folder_paths  # ← now a list

        
    def preprocessing_text(self,text):
        """
        Function to preprocess text to feature array.
        Parameters:
            - text : str, input text to be processed
        Returns:
            - text_feat_arr : np.array, text feature array
        """
        text_feat_arr = self.model.get_text_features(text)
        # tokenize then embed the text and convert to numpy array 
        text_feat_arr = text_feat_arr.reshape(1, -1).astype('float32') #=> float32
        return text_feat_arr

    def preprocessing_image(self, image_path):

        """
        Function to preprocess image to feature array.
        Parameters:
            - image_path : str, path to the image
        Returns:
            - image_feat_arr : np.array, image feature array
        """
        image_feat_arr = self.model.get_image_features(image_path)
        # tokenize then embed the text and convert to numpy array 
        image_feat_arr = image_feat_arr.reshape(1, -1).astype('float32') #=> float32
        return image_feat_arr

    def predict_func(self, feat_arr, del_keyframes=False):
        """
        Function to perform search.
        Parameters:
            - feat_arr : np.array, feature array
        Returns:
            - search_results : list, search results
        """
        k = 600
        D, I = self.faiss_db.search(feat_arr, k)  # Perform the FAISS search
        search_results = []
        for instance in zip(I[0], D[0]):
            ins_id, distance = instance
            video_name, idx = self.db[ins_id]
            video_name = video_name.split('/')[-1]
            
            #frames_folder = f"{self.keyframe_folder_paths}/Keyframes_{video_name[:3]}/keyframes/{video_name}"
            
            # Lọc L26 nếu tick
            if del_keyframes and video_name.upper().startswith("L26"):
                continue
        
            # Tìm folder chứa video
            frames_folder = None
            for folder in self.keyframe_folder_paths:
                candidate = os.path.join(folder, f"Keyframes_{video_name[:3]}/keyframes/{video_name}")
                if os.path.exists(candidate):
                    frames_folder = candidate
                    break

            if frames_folder is None:
                continue  # Nếu không tìm thấy folder nào, bỏ qua kết quả
            
            keyframe_id = sorted(os.listdir(frames_folder))[idx].split('.')[0]
            video_name = video_name + '.mp4'
            result = {
                "video_name": str(video_name),
                "keyframe_id": str(keyframe_id),
                "score": str(distance)
            }
            search_results.append(result)
        return search_results


    def search_text(self,text, del_keyframes=False):
        """
        Function for text search.
        Parameters:
            - text : str, input text to be processed
        Returns:
            - search_results : list, search results
        """
        # Preprocess text
        feat_arr = self.preprocessing_text(text)
        # Perform search
        search_results = self.predict_func(feat_arr, del_keyframes=del_keyframes)
        return search_results

    def search_image(self,image_path, del_keyframes=False):
        """
        Function for image search.
        Parameters:
            - image_path : str, path to the image
        Returns:
            - search_results : list, search results
        """
        # Preprocess image
        # print(image_path)
        # folder_image = "/".join(image_path.split('/')[-2:])
        # print(folder_image)
        # image_path = '/dataset/AIC_2025/SIU_Pumpking/0/frames/autoshot' + folder_image
        # print(image_path)
        feat_arr = self.preprocessing_image(image_path)
        # Perform search
        search_results = self.predict_func(feat_arr, del_keyframes=del_keyframes)
        return search_results

        


