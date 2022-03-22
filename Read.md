# Install Instructions
## Docker
I am programming on an apple m1 arm based machine. Due to this, I had to build a docker image `tf-devel-cpu-arm64v8-jupyter` for tensorflow that is based off of Arm. I did this buy downloading the tensorflow repository and running docker build for the the `arm64v8` dockerfile. This could probably be automated with a bash script, but your cpu architecture may change. Update the base docker image to a tensorflow base image supporte by your pc.

## Environment paths
I had to set my LD_PRELOAD path for torch with the following:

```
  export LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/torch/lib/libgomp-d22c30c5.so.1
```

## Install TA-LIB
Download tar file and run `make` then `make install`