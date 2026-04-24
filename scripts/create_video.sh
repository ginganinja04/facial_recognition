ffmpeg -framerate 2 -pattern_type glob -i "data/tracks_visualized/balcony/*.jpg" \
-c:v libx264 -pix_fmt yuv420p balcony.mp4

ffmpeg -framerate 2 -pattern_type glob -i "data/tracks_visualized/bar_stage/*.jpg" \
-c:v libx264 -pix_fmt yuv420p bar_stage.mp4

ffmpeg -framerate 2 -pattern_type glob -i "data/tracks_visualized/inside_bar/*.jpg" \
-c:v libx264 -pix_fmt yuv420p inside_bar.mp4

ffmpeg -framerate 2 -pattern_type glob -i "data/tracks_visualized/street_view/*.jpg" \
-c:v libx264 -pix_fmt yuv420p street_view.mp4