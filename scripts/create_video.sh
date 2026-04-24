ffmpeg -framerate 2 -pattern_type glob -i "mini_demo2/data/tracks_visualized/street_view/day1/*.jpg" \
-c:v libx264 -pix_fmt yuv420p output.mp4