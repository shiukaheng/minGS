from gs.core.GaussianModel import GaussianModel
from gs.io.colmap import load
from gs.trainers.basic import train

# Load COLMAP dataset
cameras, pointcloud = load('your_dataset_path/')

# Initialize Gaussian model
model = GaussianModel.from_point_cloud(pointcloud).cuda()

# Train the model
train(model, cameras, iterations=4000, densify_until_iter=3000)