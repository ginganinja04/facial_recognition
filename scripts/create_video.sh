ffmpeg -framerate 2 -pattern_type glob -i "*.jpg" \
-c:v libx264 -pix_fmt yuv420p output.mp4