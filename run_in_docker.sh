#!/bin/bash

command=$@
image="gltorch_tensorflow:yatao"
#image="gltorch:face_reenact_GDPW"
tmp_dir=/app/tmp/$HOSTNAME
TENSORFLOW_CUDA_CACHE="-e NVDIFFRAST_CACHE_DIR=$tmp_dir"

echo "Using container image: $image"
echo "Running command: $command"

docker run \
    --name face_reenact_GDPW \
    --rm -it --gpus all --user $(id -u):$(id -g) \
    -v `pwd`:/app \
    -v /home/$USER/.cache:/.cache \
    --workdir /app \
    --shm-size 512m \
    -e USER=$USER -e TORCH_EXTENSIONS_DIR=$tmp_dir $TENSORFLOW_CUDA_CACHE \
    $image $command

# use 'docker exec -it <container_name_or_id> /bin/bash' to open another terminal for the same docker container