# LeNet-5

LeNet-5 comes from < Gradient-Based Learning Applied to Document Recognition > [1998]

![](https://cuijiahua.com/wp-content/uploads/2018/01/dl_3_4.jpg)

1. Input Layer
   - 32 x 32

2. C1 Layer
   - Input Size : 32 x 32 x 1
   - Kernel Size : 5 x 5
   - Kernel Num : 6
   - Kernel Stride : 1
   - Kernel Padding : Valid
   - Output Size : 28 x 28 x  6
   - Trainable Parameters : (5 * 5 + 1) * 6
   
3. S2 Layer
   - Input Size : 28 x 28 x 6
   - Kernel Size : 2 x 2
   - Kernel Stride : 2
   - Output Size : 14 x 14 x 6
   
4. C3 Layer
   - Input Size : 14 x 14 x 6
   - Kernel Size : 5 x 5
   - Kernel Num : 16 ( 6 + 6 + 3 + 1)
   - Kernel Stride : 1
   - Kernel Padding : Valid
   - Output Size : 10 x 10 x 16
   - Trainable Parameters : 6 * (3 * 5 * 5 + 1) + 6 * (4 * 5 * 5 + 1) + 3 * (4 * 5 * 5 + 1) + 1 * (6 * 5 * 5 + 1) 

![](https://cuijiahua.com/wp-content/uploads/2018/01/dl_3_5.png)

5. S4 Layer
   - Input Size : 10 x 10 x 16
   - Kernel Size : 2 x 2
   - Kernel Stride : 2
   - Output Size : 5 x 5 x 16
   
6. C5 Layer
   - Input Size : 5 x 5 x 16
   - Kernel Size : 5 x 5
   - Kernel Num : 120
   - Output Size : 1 x 1 x 120
   - Trainable Parameters : 120 * (16 * 5 * 5 + 1)

7. F6 Layer
   - Input Size : 1 x 1 x 120
   - Output Size : 1 x 1 x 84

8. Output Layer
   - Input Size : 1 x 1 x 84
   - Output Size : 1 x 1 x 10
