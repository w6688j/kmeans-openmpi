# Algorithm Details
K-means is an iterative clustering algorithm, which returns the cluster center given data and #clusters. Openmpi is a message parsing library used for parallel implementations. K-means minimizes the square loss between cluster center and each point belonging to that cluster. We shared the data between processes evenly and decided to communicate the errors to each process at each iteration, such that at the end every process calculates the new clusters. Doing that we have a communication cost in the order of O(cp), where c is total number of clusters and p is total number of processes. The algorithm works as follows:

1. Range based data partition. Each process reads its own part from the file system.
2. P0 picks initial centers by sampling random point from the data.
3. For 1:N
    a. Each process calculate closest cluster center for each point and accumulates the residual for each center. We hold the sum of residuals and count of points belonging to that cluster separately to reduce the communication and to be be able to calculate the new centers effortlessly .
    b. Share residual sum & counts by using `AllReduce`. 
    c. Each process calculates new centers by summing the individual residual sums together and dividing the result by the total count of that cluster (number of points belonging to that cluster)

# Running mpi-code 
We wrote a python script to generate random points sampled from normal distribution around 9 different points with 0.5 standard deviation. Visualization of a small dataset sampled provided below. The centers are at {(1,-1,0), (1,1,0),(-1,-1,0),(-1,1,0),(0,0,3),(1,-1,6), (1,1,6),(-1,-1,6),(-1,1,6)}.

![points](images/points.png) 

```python
python create_data.py 10000 points.dat
```

After running `make` to compile c-code one can run openmpi code with
```
mpirun -np 4 ./k_means 3 10000 9 points.dat 30
#3d data,Total: 10000 points,2500 points per process from file points.dat
#Center 0 from line 73
#...
#centers:
#-0.967202 0.981146 -0.044941 
#....
#Time elapsed is 0.022455 seconds.
```

where `k_means <dimension of a point> <N> <#clusters> <datafile> <#iterations>`. 

# Results
I run the openMPI code on the Raspberry Pi-cluster we created and on some big clusters like NYU’s Prince and TACC’s Stampede. I investigated weak and strong scaling on up to 256 nodes. Pi-cluster doesn’t show good weak scaling (from 4 process to 8) and we believe that this behavior is due to the shared resources. From 8 process to 16 pi scales perfectly. K-means scales almost perfectly in Stampede and Prince. 

| Weak Scaling | Strong Scaling |
| ------------ | -------------- |
| ![weak1](images/weak1.png) | ![strong1](images/strong1.png)|

To be able observe the scaling behaviour better between processes we can normalize the timings by dividing it to their mean value. 

| Weak Scaling | Strong Scaling |
| ------------ | -------------- |
| ![weak2](images/weak2.png) | ![strong2](images/strong2.png)|  

#### 单进程
3维，10000个点，9个中心，10000次
```asm
不成功
编译: gcc k_means_single.c -o k_means_single -lrt -lm 
运行: ./k_means_single 3 10000 9 points.dat 1000

用mpi的单进程运行
运行: mpirun -np 1 ./k_means 3 10000 9 points.dat 10000
结果: 
3d data,Total: 10000 points,10000 points per process from file points.dat
Center 0 from line 540
Center 1 from line 2567
Center 2 from line 3426
Center 3 from line 3926
Center 4 from line 5211
Center 5 from line 5368
Center 6 from line 5736
Center 7 from line 7763
Center 8 from line 9172
centers:
-0.966626 0.992238 -0.044355 
1.061053 0.962075 0.031377 
0.949319 -1.079107 0.031815 
-1.077819 -0.999596 0.064136 
0.026441 0.036646 3.204902 
1.044101 1.012886 5.994005 
-0.991692 0.987134 6.008131 
-1.046635 -1.001105 6.027964 
1.031660 -1.074799 5.995452 
Time elapsed is 64.095202 seconds.
```

#### 多进程
```asm
编译: gcc k_means.c -o k_means_single -lrt -lm 
``` 
##### p=2
```asm
运行: mpirun -np 2 ./k_means 3 10000 9 points.dat 10000
结果:
3d data,Total: 10000 points,5000 points per process from file points.dat
Center 0 from line 540
Center 1 from line 2567
Center 2 from line 3426
Center 3 from line 3926
Center 4 from line 5211
Center 5 from line 5368
Center 6 from line 5736
Center 7 from line 7763
Center 8 from line 9172
centers:
-0.966626 0.992238 -0.044355 
1.061053 0.962075 0.031377 
0.949319 -1.079107 0.031815 
-1.077819 -0.999596 0.064136 
0.026441 0.036646 3.204902 
1.044101 1.012886 5.994005 
-0.991692 0.987134 6.008131 
-1.046635 -1.001105 6.027964 
1.031660 -1.074799 5.995452 
Time elapsed is 33.216177 seconds.
```
##### p=4
```asm
运行: mpirun -np 4 ./k_means 3 10000 9 points.dat 10000
结果:
3d data,Total: 10000 points,2500 points per process from file points.dat
Center 0 from line 59
Center 1 from line 540
Center 2 from line 3426
Center 3 from line 3926
Center 4 from line 5211
Center 5 from line 5368
Center 6 from line 5736
Center 7 from line 7763
Center 8 from line 9172
centers:
-0.966820 0.986599 -0.044579 
-1.032727 -1.011959 0.067240 
1.059765 0.976034 0.033215 
0.997752 -1.061490 0.022291 
0.026519 0.035481 3.202765 
1.044101 1.012886 5.994005 
-0.991692 0.987134 6.008131 
-1.046635 -1.001105 6.027964 
1.031660 -1.074799 5.995452 
Time elapsed is 16.974430 seconds.
```
##### p=8
```asm
运行: mpirun -np 8 ./k_means 3 10000 9 points.dat 10000
结果:
3d data,Total: 10000 points,1250 points per process from file points.dat
Center 0 from line 59
Center 1 from line 540
Center 2 from line 3426
Center 3 from line 3926
Center 4 from line 5211
Center 5 from line 5368
Center 6 from line 5736
Center 7 from line 7763
Center 8 from line 9172
centers:
0.303797 81379.183544 -0.430380 
0.678788 30419.854545 -0.436364 
0.408451 69200.612676 -0.450704 
-0.140151 20.884488 2.543016 
0.723881 19568.738806 -0.492537 
0.561404 55720.035088 -0.421053 
0.247863 94435.145299 -0.427350 
0.742647 9656.485294 -0.389706 
0.538961 42230.103896 -0.500000 
Time elapsed is 53.633947 seconds.
```
##### p=16
```asm
运行: mpirun -np 16 ./k_means 3 10000 9 points.dat 10000
结果:
3d data,Total: 10000 points,625 points per process from file points.dat
Center 0 from line 59
Center 1 from line 540
Center 2 from line 3426
Center 3 from line 3926
Center 4 from line 5211
Center 5 from line 5736
Center 6 from line 7763
Center 7 from line 8690
Center 8 from line 9172
centers:
0.617886 53724.097561 -0.243902 
0.674074 32833.422222 -0.370370 
0.333333 73745.869919 -0.373984 
0.764706 15668.068627 -0.359447 
-0.075764 43.224037 3.293977 
0.200000 91046.764706 -0.247059 
0.842248 -0.323529 876.058824 
0.967153 0.000000 7183.166667 
1.119455 0.538462 3540.384615 
Time elapsed is 53.560410 seconds.
```