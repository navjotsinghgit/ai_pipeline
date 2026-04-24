from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# download the model
model_path = hf_hub_download(
    repo_id="macpaw-research/yolov11l-ui-elements-detection",
    filename="ui-elements-detection.pt",
)

# load and run prediction
model = YOLO(model_path)
results = model.predict("ai_pipeline/detection_service/2019-12-12T152726.jpg")

# display result
results[0].show()
