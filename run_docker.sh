docker run --gpus all -it \
  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/home/evs/forSR_test:/home/appuser/forSR_test" \
  --name=mnist-gui-instance mnist_gui