"""

# Copy the model files from a running container to the host
docker cp <container_id>:/path/to/model/files /host/path/
Mount Host Directories:
You can also set up Docker to mount host directories into the container. This way, any changes made inside the container (such as saving the trained model files) will be reflected on the host. This method allows you to access the trained model files directly from the host machine.

bash
Copy code
# Run a container with a mounted host directory
docker run -v /host/path:/container/path -it my_ml_container
Save Model Files to Shared Volume:
If you have a shared Docker volume defined, you can save the trained model files to that volume within the container. The changes will be accessible on the host and can be shared between multiple containers.

bash
Copy code
# Run a container with a shared volume
docker run -v my_shared_volume:/container/path -it my_ml_container




sudo apt install nvidia-cuda-toolkit
sudo apt install g++

/media/local-data/Cuong/YCB-Video-Dataset/bop_ycbv/bop_datasets/ycbv

"""
import cv2

img = cv2.imread('/home/ana/Study/CVPR/idisc/ycbv_dataset/test/000054/rgb/001694.png')
cv2.imshow('iamg', img)
cv2.waitKey(0)