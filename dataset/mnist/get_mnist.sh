#!/bin/sh

download="true"
text="true"
convert="false"

if [ $download = "true" ]; then
# download dataset
	echo 'Downloading dataset...'
	curl http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz > train-images-idx3-ubyte.gz
	curl http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte.gz
	curl http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz > t10k-images-idx3-ubyte.gz
	curl http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz > t10k-labels-idx1-ubyte.gz

# decompress dataset
	echo 'Decompressing...'
	gzip -dc train-images-idx3-ubyte.gz > train-images-idx3-ubyte
	gzip -dc train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte
	gzip -dc t10k-images-idx3-ubyte.gz  > t10k-images-idx3-ubyte
	gzip -dc t10k-labels-idx1-ubyte.gz  > t10k-labels-idx1-ubyte
fi

# convert to text file
if [ $text = "true" ]; then
	echo 'Converting to text file'
	# IF GNU od command only
	#od -An -v -tu1 -j16 -w784 train-images-idx3-ubyte | sed -e 's/^ *//' | tr -s ' ' > train-images.txt
	#od -An -v -tu1 -j8 -w1 train-labels-idx1-ubyte | tr -d ' ' > train-labels.txt
	#od -An -v -tu1 -j16 -w784 t10k-images-idx3-ubyte | sed -e 's/^ *//' | tr -s ' ' > test-images.txt
	#od -An -v -tu1 -j8 -w1 t10k-labels-idx1-ubyte | tr -d ' ' > test-labels.txt
	# for any od command
	od -An -v -tu1 -j16 train-images-idx3-ubyte | tr '\n' ' ' | tr -s ' ' | awk 'BEGIN { RS=" "; i=0 } { if(i == 784) { i=0; printf "\n" }; printf $1; printf " "; i++ }' > train-images.txt
	od -An -v -tu1 -j8 train-labels-idx1-ubyte | tr '\n' ' ' | tr -d ' ' | fold -s -w 1 > train-labels.txt
	od -An -v -tu1 -j16 t10k-images-idx3-ubyte | tr '\n' ' ' | tr -s ' ' | awk 'BEGIN { RS=" "; i=0 } { if(i == 784) { i=0; printf "\n" }; printf $1; printf " "; i++ }' > test-images.txt
	od -An -v -tu1 -j8 t10k-labels-idx1-ubyte | tr '\n' ' ' | tr -d ' ' | fold -s -w 1 > test-labels.txt
fi

# convert to imagefile
if [ $convert = "true" ]; then
	echo 'Converting to image file'
	count=0
	cat train-images.txt | while read line; do
		echo $line | awk '
			BEGIN { print "P2 28 28 255" }
			{ for (i = 1; i <= NF; i++) printf("%d%s", $i, i % 14 ? " " : "\n") }' |\
		pnmtopng - > $(printf "mnist/image%05d.png" $count)
		count=$(expr $count + 1)
	done
fi
