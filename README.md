# littleCNN  
Convolutional Neural Network  
for simple, easily.  

## features  

## Usage  
First, clone this repository.  
```shell
git clone https://github.com/satoshishimada/littleCNN.git
cd littleCNN
```

### Samples  
* #### In Windows7  
 First, cmake by `CMakeList.txt` file.  
 Second, click `ALL_BUILD.vcxproj`, then open Visual Studio.
 Click to Build -> Build Solution(F7).
 Finally, execute executable file in littleCNN directory.
 
* #### In Linux  

 * logic  
  compile sample.  
  ```shell
  make logic
  ```

 * mnist  
  download dataset.  
  ```shell
  cd dataset/mnist
  ./get_mnist.sh
  cd ../../
  ```
  compile sample.
  ```shell
  make mnist
  ```

## Author  
Satoshi SHIMADA  
contact: email mylinux1204@gmail.com  
