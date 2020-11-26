#!/bin/bash

docker run -it --rm -v "$PWD":/usr/src/app pp-sim "$@"
