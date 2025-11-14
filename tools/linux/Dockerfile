FROM docker.io/buildpack-deps:jammy AS build-casparcg
ADD tools/linux/install-dependencies /

RUN apt-get update && /install-dependencies

RUN mkdir /source && mkdir /build && mkdir /install

COPY ./src /source

WORKDIR /build

ARG CC
ARG CXX
ARG GIT_HASH

RUN cmake -GNinja /source -DUSE_STATIC_BOOST=ON -DUSE_SYSTEM_CEF=OFF

RUN cmake --build .

RUN cmake --install . --prefix staging

# Find a better way to copy deps
RUN ln -s /build/staging /staging && \
	/source/shell/copy_deps.sh /staging/bin/casparcg /staging/lib

# Download and extract media-scanner
FROM build-casparcg AS download-scanner
RUN apt-get update && apt-get install -y curl jq unzip
RUN mkdir -p /media-scanner && cd /media-scanner && \
	LATEST_RELEASE=$(curl -s https://api.github.com/repos/casparcg/media-scanner/releases/latest | jq -r '.assets[] | select(.name | contains("linux-x64")) | .browser_download_url') && \
	curl -L "$LATEST_RELEASE" -o scanner.zip && \
	unzip scanner.zip && \
	rm scanner.zip && \
	chmod +x scanner

FROM docker.io/nvidia/opengl:1.2-glvnd-devel-ubuntu22.04
COPY --from=build-casparcg /staging /opt/casparcg
COPY --from=download-scanner /media-scanner /opt/casparcg/scanner

RUN set -ex; \
	apt-get update; \
	DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
	tzdata \
	vim \
	libc++1 \
	libnss3 \
	fontconfig \
	libatk1.0-0 \
	libcups2 \
	libxcomposite1 \
	libxdamage1 \
	libatk-bridge2.0-0 \
	xvfb \
	libegl1 \
	libgl1 \
	libgles2 \
	libopengl0 \
	libvulkan1 \
	vulkan-tools \
	ffmpeg \
	; \
	rm -rf /var/lib/apt/lists/*

WORKDIR /opt/casparcg
RUN chmod 755 run.sh
ADD tools/linux/run_docker.sh ./
ADD tools/linux/run_docker.sh ./

CMD ["./run_docker.sh"]
