FROM nvcr.io/nvidia/tensorrt:24.01-py3

# install packages for opencv
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git git-lfs pkg-config libgtk-3-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
  libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev \
  python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-dev \
  sudo libgflags-dev libboost-dev libxml2-dev \
  libyaml-cpp-dev sqlite3 libsqlite3-dev libboost-all-dev \
  lsb-release peco feh fim openssh-server tmux curl
RUN apt-get -y clean && rm -rf /var/lib/apt/lists/*

#Build opencv cuda
#! edit CUDA_ARCH_VERSION to match your GPU
ARG CUDA_ARCH_VERSION=8.7
RUN mkdir /root/opencv_build
WORKDIR /root/opencv_build/
RUN git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git && \
  cd /root/opencv_build/opencv && git checkout 4.10.0 && \
  cd /root/opencv_build/opencv_contrib && git checkout 4.10.0 && \
  cd /root/opencv_build/opencv && mkdir build

WORKDIR /root/opencv_build/opencv/build
RUN cmake -G Ninja -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D BUILD_opencv_cudacodec=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUDA=ON \
-D WITH_CUBLAS=1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D BUILD_EXAMPLES=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=${CUDA_ARCH_VERSION} \
..
RUN ninja && ninja install
WORKDIR /root
RUN rm -rf opencv_build

# install netron to show model structure
RUN pip install netron
