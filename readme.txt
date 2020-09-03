Outputs can be found in subfolders: subtask1, subtask2, subtask3 already but can be recalculated by following the instructions below

To compile and run subtask 3 on dart5.jpg (or dart0.jpg - dart15.jpg):
g++ -o dartHoughViola dartHoughViola.cpp /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4
./dartHoughViola dart5.jpg

To compile and run subtask 2 on dart5.jpg (or dart0.jpg - dart15.jpg):
g++ -o dart dart.cpp /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4
./dart dart5.jpg

To compile and run subtask 1 on dart5.jpg (or dart4.jpg, dart13, dart14.jpg, dart15.jpg):
g++ -o face face.cpp /usr/lib64/libopencv_core.so.2.4 /usr/lib64/libopencv_highgui.so.2.4 /usr/lib64/libopencv_imgproc.so.2.4 /usr/lib64/libopencv_objdetect.so.2.4
./face dart5.jpg
