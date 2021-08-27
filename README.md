# U-net

This repository contains a custom implementation of the U-net architecture using Pytorch. The U-net first appeared in a paper by [Ronneberger](https://arxiv.org/abs/1505.04597) et al. In the abstract; they describe the network architecture as a "contracting path to capture context and a symmetric expanding path that enables precise localization." The network consists of 3x3 convolutions followed by max-pooling to encode or contract the input. The expansion occurs using transposed convolutions. The purpose of this network when introduced was image segmentation. In this project, the U-net completed both denoising (regression) and segmentation of computed tomography (CT) images of bone.

For both tasks, the network layers were adjusted to accommodate a larger image size and still run on the GPU. In these examples, the size of the layers used was halved (32, 64, 128, 256, and 512). The input to the network was a 3-dimensional image stack consisting of 5 slices. Since CT data is sequential, this dimensionality added more structural context for the U-net. The network used the 3D input to produce a single 2D slice as output.

<br></br>

## Denoising

Denoising or regression of CT images is interesting because it enables faster acquisition times with less radiation exposure. Images to train the network consisted of acquiring two scans of the same sample. A low-quality (noisy) scan was obtained for input, and a high-quality scan was acquired as the target. The pairing of these images resulted in a supervised dataset. The U-net learned to reduce noise from the input image by minimizing the MSE between the network output and the target images. Below is an example of the denoised image after only ten epochs of training. 

<table>
  <tr>
    <td> <b>Input</b> </td>
    <td> <b>Target</b> </td>
    <td> <b>U-net Output</b> </td>
  </tr>
  <tr>
    <td> <img src="images/denoise_01_input.png" width=290px> </td>
    <td> <img src="images/denoise_02_target.png" width=290px> </td>
    <td> <img src="images/denoise_03_unet.png" width=290px> </td>
  </tr>
  <caption></caption>
 </table>
 
 <br></br>
 
 ## Segmentation
 
The segmentation task in this project was to localize the crack within the bone. Segmenting the rack by hand is a very time-consuming process, so the ability to segment using the network is very appealing. The training dataset consisted of denoised images as input and images comprised of a manual crack segmentation. The loss function in this application was cross-entropy. The network identified two classes: crack and not crack. Below is an example of the segmentation performance of the U-net after only ten epochs. 
 
 <table>
  <tr>
    <td> <b>Input</b> </td>
    <td> <b>Target</b> </td>
    <td> <b>U-net Output</b> </td>
  </tr>
  <tr>
    <td> <img src="images/segment_01_input.png" width=290px> </td>
    <td> <img src="images/segment_02_target.png" width=290px> </td>
    <td> <img src="images/segment_03_unet.png" width=290px> </td>
  </tr>
 </table>
 
 <br></br>
 
The network performed very well when there was a crack in the image. At times the network would identify a crack when there was no crack present. This misclassification generally occurred when a natural feature in the bone had a similar orientation and geometry as some of the cracks. Below is an example of one of these occurrences.
 
 <table>
  <tr>
    <td> <b>Input</b> </td>
    <td> <b>Target</b> </td>
    <td> <b>U-net Output</b> </td>
  </tr>
  <tr>
    <td> <img src="images/crack2_01_input.png" width=290px> </td>
    <td> <img src="images/crack2_02_target.png" width=290px> </td>
    <td> <img src="images/crack2_03_unet.png" width=290px> </td>
  </tr>
 </table>
  
