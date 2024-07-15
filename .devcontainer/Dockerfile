FROM nvcr.io/nvidia/tensorrt:24.01-py3

# install packages for opencv
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git git-lfs pkg-config libgtk-3-dev \
  libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
  libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev \
  python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-dev
RUN apt-get -y clean && rm -rf /var/lib/apt/lists/*

#Build opencv4.5.5
RUN mkdir /root/opencv_build
WORKDIR /root/opencv_build/
RUN git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git && \
  cd /root/opencv_build/opencv && git checkout 4.5.5 && \
  cd /root/opencv_build/opencv_contrib && git checkout 4.5.5 && \
  cd /root/opencv_build/opencv && mkdir build

WORKDIR /root/opencv_build/opencv/build
RUN cmake -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D WITH_GSTREAMER=ON \
  -D WITH_LIBV4L=ON \
  -D BUILD_opencv_python2=ON \
  -D BUILD_opencv_python3=ON \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
  -D CMAKE_INSTALL_PREFIX=/usr/local ..
RUN make -j$(nproc) && make install
WORKDIR /root
RUN rm -rf opencv_build

# install packages that are used for trt inferences
RUN apt-get update && apt install -y --no-install-recommends \
  sudo libgflags-dev libboost-dev libxml2-dev \
  libyaml-cpp-dev sqlite3 libsqlite3-dev libboost-all-dev \
  lsb-release peco feh fim openssh-server tmux curl
RUN apt-get -y clean && rm -rf /var/lib/apt/lists/*

# install netron to show model structure
RUN pip install netron

# copy 
