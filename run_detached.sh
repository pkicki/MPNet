docker run -itd\
    --runtime=nvidia \
    --env="CONTAINER_NAME" \
    --env="DISPLAY" \
    --env="NVIDIA_VISIBLE_DEVICES=all" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="PYTHONPATH=/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages:/opt/openrobots/lib/python3.8/site-packages:/home/user/miniconda/lib/python3.8/site-packages"\
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
    --volume="/home/piotr/b8/MPNet:/airhockey" \
    --privileged \
    --network=host \
    --name="torch"\
    torch
    

