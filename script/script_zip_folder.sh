# first navigate into folder.
for i in */; do zip -0 -r "${i%/}.zip" "$i" & done; wait