docker run --rm --name media-scanner -d --network host \
  -v "$(pwd)/casparcg.config:/usr/src/app/casparcg.config:ro" \
  -v "$(pwd)/media:/usr/src/app/media:ro" \
  -v "$(pwd)/template:/usr/src/app/template:ro" \
  media-scanner:latest