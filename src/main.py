from src.pipeline import SafetyPipeline


pipeline = SafetyPipeline(
    source="data/sample_videos/output.mp4",
    ppe_model_path="models/best.pt",
    pose_model_path="yolo26s-pose.pt",  # downloads automatically first run
)

# Define a restricted zone (adjust coordinates to your factory layout)
pipeline.add_zone("RESTRICTED_A", [
    [100, 400], [300, 400], [300, 600], [100, 600]
])

pipeline.run(display=True)