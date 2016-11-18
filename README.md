# littleCNN  
The Convolutional Neural Network library.  
for simple, easily.  

## features  
* Training with simple dataset
* Calculate test accuracy in training
* Save/Load parameters
* Layers
 * Fully connected layer
 * Convolutional layer
 * Max-Pooling layer

## Usage  
Clone this repository.  
And change working directory.  
```shell
git clone https://github.com/satoshishimada/littleCNN.git
cd littleCNN
```

### Samples  
* #### In Windows(7)  
 First, cmake by `CMakeList.txt` file.  
 Second, click `ALL_BUILD.vcxproj`, then open Visual Studio.
 Click to Build -> Build Solution(F7).  
 Finally, execute executable file in littleCNN directory.
 
* #### In Linux  

 * logic  
  Compile and run sample.
  ```shell
  make logic
  mkdir -p parameters/logic
  ./logic
  ```

 * mnist  
  Download dataset.  
  Then, compile and run sample.
  ```shell
  cd dataset/mnist
  ./get_mnist.sh
  cd ../../
  make mnist
  mkdir -p parameters/mnist
  ./mnist
  ```

## Author  
Satoshi SHIMADA  
contact: email mylinux1204@gmail.com  

