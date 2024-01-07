file='sift_warp.cpp'
g++ -g $file -o out `pkg-config --cflags --libs opencv`
./out 
rm ./out