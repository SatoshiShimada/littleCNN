#!/bin/sh

download="true"
text="true"
convert="false"

if [ $download = "true" ]; then
# download dataset
	wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
	wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# decompress dataset
gzip -dc train-images-idx3-ubyte.gz > train-images-idx3-ubyte
gzip -dc train-labels-idx1-ubyte.gz > train-labels-idx1-ubyte
gzip -dc t10k-images-idx3-ubyte.gz  > t10k-images-idx3-ubyte
gzip -dc t10k-labels-idx1-ubyte.gz  > t10k-labels-idx1-ubyte
fi

# convert to text file
if [ $text = "true" ]; then
	od -An -v -tu1 -j16 -w784 train-images-idx3-ubyte | sed -e 's/~ *//' | tr -s ' ' > train-images.txt
	od -An -v -tu1 -j8 -w1 train-labels-idx1-ubyte | tr -d ' ' > train-labels.txt
	od -An -v -tu1 -j16 -w784 t10k-images-idx3-ubyte | sed -e 's/~ *//' | tr -s ' ' > test-images.txt
	od -An -v -tu1 -j8 -w1 t10k-labels-idx1-ubyte | tr -d ' ' > test-labels.txt
fi

# convert to imagefile
if [ $convert = "true" ]; then
	count=0
	cat train-images.txt | while read line; do
		echo $line | awk '
			BEGIN { print "P2 28 28 255" }
			{ for (i = 1; i <= NF; i++) printf("%d%s", $i, i % 14 ? " " : "\n") }' |\
		pnmtopng - > $(printf "mnist/image%05d.png" $count)
		count=$(expr $count + 1)
	done
fi

