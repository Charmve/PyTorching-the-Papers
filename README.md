# PyTorching the Papers
Study state-of-the-art papers with code 💪

<!--
## StegaStamp: Invisible Hyperlinks in Physical Photographs
<p align="center">
  <a href="https://www.youtube.com/watch?v=E8OqgNDBGO0" target="_blank">
    <img src="https://res.cloudinary.com/marcomontalbano/image/upload/v1601457995/video_to_markdown/images/youtube--E8OqgNDBGO0-c05b58ac6eb4c4700831b2b3070cd403.jpg" alt="StegaStamp: Invisible Hyperlinks in Physical Photographs" width="426" height="240" />
  </a>
</p>
-->

<table>
<tr>
	<td><font size="4">6.</font></td>
	<td><center><a href="https://www.youtube.com/watch?v=yvNZGDZC3F8" target="_blank"><img src="https://img-blog.csdnimg.cn/img_convert/53252d7269b7a184bab9824d5e039787.png" width="820" height="%100"/></a></center>
	</td>
	<td>
		<p align="center"><font size="5"><b>PointRend</b>: Image Segmentation as Rendering</font>
			<br><b>Alexander Kirillov<sup>∗</sup></b>, Yuxin Wu, Kaiming He, Ross Girshick
		</p>
		<p align="left">
	        <font size =2> <b>Abstract</b>. We present a new method for efficient high-quality image segmentation of objects and scenes. By analogizing classical computer graphics methods for efficient rendering with over- and undersampling challenges faced in pixel labeling tasks, we develop a unique perspective of image segmentation as a rendering problem. From this vantage, we present the PointRend (Point-based Rendering) neural network module: a module that performs point-based segmentation predictions<details>
  		    <summary>Click to expand</summary>
			at adaptively selected locations based on an iterative subdivision algorithm. PointRend can be flexibly applied to both instance and semantic segmentation tasks by building on top of existing state-of-the-art models. While many concrete implementations of the general idea are possible, we show that a simple design already achieves excellent results. Qualitatively, PointRend outputs crisp object boundaries in regions that are over-smoothed by previous methods. Quantitatively, PointRend yields significant gains on COCO and Cityscapes, for both instance and semantic segmentation. PointRend's efficiency enables output resolutions that are otherwise impractical in terms of memory or computation compared to existing approaches. 
		</details>
	        </font>
	     	</p>
		<p align="center"><img width="%100" height="30" src="https://cvpr2019.thecvf.com/images/sponsors/cvf_.jpg"> CVPR 2020
			<br>[<b><a href="https://arxiv.org/abs/1912.08193" target="_blank">arXiv</a></b> | <a href="https://www.youtube.com/watch?v=yvNZGDZC3F8" target="_blank"><b>Video</b></a>  | <a href="https://github.com/facebookresearch/detectron2/tree/master/projects" target="_blank"><b>Project</b></a> | <a href="https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend" target="_blank"><b>Code</b></a>]
		</p>
	</td>
</tr>
<tr>
	<td><font size="4">5.</font></td>
	<td><center><a href="https://www.youtube.com/watch?v=sysySMr3YN4" target="_blank"><img src="https://img-blog.csdnimg.cn/img_convert/3dd379d7f61591dc731adfcdf032b91b.png" width="820" height="%100" /></a></center>
	</td>
	<td>
		<p align="center"><font size="5"><b>PolarMask</b>: Single Shot Instance Segmentation with Polar Representation</font>
			<br><b>Enze Xie<sup>∗</sup></b>, Peize Sun, Xiaoge Song, Wenhai Wang, Ding Liang, Chunhua Shen, Ping Luo
		</p>
		<p align="left">
	        <font size =2> <b>Abstract</b>. In this paper, we introduce an anchor-box free and single shot instance segmentation method, which is conceptually simple, fully convolutional and can be used as a mask prediction module for instance segmentation, by easily embedding it into most off-the-shelf detection methods. Our method, termed PolarMask, formulates the instance segmentation problem as instance center classification and dense distance regression in a polar coordinate. Moreover, we propose two effective approaches to deal with sampling high-quality center examples and optimization for dense distance regression, respectively, which can significantly improve the performance and simplify the training process. <details>
  		    <summary>Click to expand</summary>Without any bells and whistles, PolarMask achieves 32.9% in mask mAP with single-model and single-scale training/testing on challenging COCO dataset. For the first time, we demonstrate a much simpler and flexible instance segmentation framework achieving competitive accuracy. We hope that the proposed PolarMask framework can serve as a fundamental and strong baseline for single shot instance segmentation tasks.
		</details>
	        </font>
	     	</p>
		<p align="center"><img width="%100" height="30" src="https://cvpr2019.thecvf.com/images/sponsors/cvf_.jpg"> CVPR 2020
			<br>[<b><a href="https://arxiv.org/abs/1909.13226" target="_blank">arXiv</a></b> | <a href="https://www.youtube.com/watch?v=sysySMr3YN4" target="_blank"><b>Video</b></a> | <a href="https://github.com/xieenze/PolarMask" target="_blank"><b>Code</b></a>  | <a href="https://zhuanlan.zhihu.com/p/84890413" target="_blank"><b>Study</b></a>]
		</p>
	</td>
</tr>
<tr>
	<td><font size="4">4.</font></td>
	<td><center><a href="https://github.com/JizhiziLi/animal-matting" target="_blank"><img src="https://github.com/JizhiziLi/animal-matting/raw/master/demo/src/sample2.jpg" alt="End-to-end Animal Image Matting" width="90" height="%100" /><img src="https://github.com/JizhiziLi/animal-matting/raw/master/demo/src/sample2.png" alt="End-to-end Animal Image Matting" width="90" height="%100" /></a></center>
	</td>
	<td>
		<p align="center"><font size="5"><b>End-to-end Animal Image Matting</b></font>
			<br><b>Jizhizi Li<sup>∗</sup></b>, Jing Zhang, Stephen J. Maybank, Dacheng Tao<sup>∗</sup>
		</p>
		<p align="left">
	        <font size =2> <b>Abstract</b>. Extracting accurate foreground animals from natural animal images benefits many downstream applications such as film production and augmented reality. However, the various appearance and furry characteristics of animals challenge existing matting methods, which usually require extra user inputs such as trimap or scribbles. To resolve these problems, we study the distinct roles of semantics and details for image matting and decompose the task into two parallel sub-tasks: high-level semantic segmentation and low-level details matting. Specifically, we propose a novel Glance and Focus Matting network (GFM), which employs a shared encoder and two separate decoders to learn both tasks in a collaborative manner for end-to-end animal image matting. <details>
  		    <summary>Click to expand</summary>Besides, we establish a novel Animal Matting dataset (AM-2k) containing 2,000 high-resolution natural animal images from 20 categories along with manually labeled alpha mattes. Furthermore, we investigate the domain gap issue between composite images and natural images systematically by conducting comprehensive analyses of various discrepancies between foreground and background images. We find that a carefully designed composition route RSSN that aims to reduce the discrepancies can lead to a better model with remarkable generalization ability. Comprehensive empirical studies on AM-2k demonstrate that GFM outperforms state-of-the-art methods and effectively reduces the generalization error.
		</details>
	        </font>
	     	</p>
		<p align="center"><img width="%100" height="30" src="https://cvpr2019.thecvf.com/images/sponsors/cvf_.jpg"> CVPR 2020
			<br>[<b><a href="https://arxiv.org/abs/2010.16188" target="_blank">arXiv</a></b> | <a href="https://github.com/JizhiziLi/animal-matting" target="_blank"><b>Project Page</b></a> |  <b>Video</b>  |  <a href="https://github.com/JizhiziLi/animal-matting" target="_blank"><b>Code</b></a> | <a href="https://arxiv.org/pdf/2009.06613.pdf" target="_blank"><b>Related Work</b>]
		</p>
	</td>
</tr>
<tr>
	<td><font size="4">3.</font></td>
	<td><center><a href="https://www.youtube.com/watch?v=E8OqgNDBGO0" target="_blank"><img src="https://mmbiz.qpic.cn/mmbiz_png/ZNdhWNib3IRB5Jh9zIic4ScbPaullYiaFviasLGiaSiaj7o2IRn5eias4rmEkhVvgJarDoypyr8fjflX5jn5C2FmydADg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="GhostNet: GhostNet: More Features from Cheap Operations" width="820" height="%100" /></a></center>
	</td>
	<td>
		<p align="center"><font size="5"><b>GhostNet</b>: GhostNet: More Features from Cheap Operations</font>
			<br><b>Kai Han</b>, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu
		</p>
		<p align="left">
	        <font size =2> <b>Abstract</b>. Deploying convolutional neural networks (CNNs) on embedded devices is difficult due to the limited memory and computation resources. The redundancy in feature maps is an important characteristic of those successful CNNs, but has rarely been investigated in neural architecture design. This paper proposes a novel Ghost module to generate more feature maps from cheap operations. Based on a set of intrinsic feature maps, we apply a series of linear transformations with cheap cost to generate many ghost feature maps that could fully reveal information underlying intrinsic features. <details>
  		    <summary>Click to expand</summary>The proposed Ghost module can be taken as a plug-and-play component to upgrade existing convolutional neural networks. Ghost bottlenecks are designed to stack Ghost modules, and then the lightweight GhostNet can be easily established. Experiments conducted on benchmarks demonstrate that the proposed Ghost module is an impressive alternative of convolution layers in baseline models, and our GhostNet can achieve higher recognition performance (e.g. 75.7% top-1 accuracy) than MobileNetV3 with similar computational cost on the ImageNet ILSVRC-2012 classification dataset.
	        </details>
		</font>
	     	</p>
		<p align="center"><img width="%100" height="30" src="https://cvpr2019.thecvf.com/images/sponsors/cvf_.jpg"> CVPR 2020
			<br>[<b><a href="https://arxiv.org/abs/1911.11907" target="_blank">arXiv</a></b> | <a href="https://github.com/huawei-noah/ghostnet" target="_blank"><b>Project Page</b></a> |  <a href="https://www.youtube.com/watch?v=xthNLbn1bUY" target="_blank"><b>Video</b></a>  |  <a href="https://github.com/huawei-noah/ghostnet" target="_blank"><b>Code</b></a>]
		</p>
	</td>
</tr>
<tr>
	<td><font size="4">2.</font></td>
	<td><center><a href="https://www.youtube.com/watch?v=E8OqgNDBGO0" target="_blank"><img src="https://uploads-ssl.webflow.com/51e0d73d83d06baa7a00000f/5ca3b3c3ca205d53d7e986a1_pipeline-01-p-2000.png" alt="StegaStamp: Invisible Hyperlinks in Physical Photographs" width="640" height="%100" /><img src="https://uploads-ssl.webflow.com/51e0d73d83d06baa7a00000f/5ca400f82e5a6c5707af7189_pipeline_train-01-p-2000.png" alt="StegaStamp: Invisible Hyperlinks in Physical Photographs" width="1022" height="%100" /></a></center>
	</td>
	<td>
		<p align="center"><font size="5"><b>StegaStamp</b>: Invisible Hyperlinks in Physical Photographs</font>
			<br><b>Matthew Tancik<sup>∗</sup></b>, Ben Mildenhall<sup>∗</sup>, Ren Ng
		</p>
		<p align="left">
	        <font size =2> <b>Abstract</b>. Printed and digitally displayed photos have the ability to hide imperceptible digital data that can be accessed throughinternet-connected imaging systems. Another way to think about this is physical photographs that have unique QR codes invisibly embedded within them. This paper presentsan architecture, algorithms, and a prototype implementation addressing this vision. Our key technical contribution is StegaStamp, a learned steganographic algorithm to enable robust encoding and decoding of arbitrary hyperlink bitstrings into photos in a manner that approaches perceptual invisibility. StegaStamp comprises a deep neural network that learns an encoding/decoding algorithm robust to image perturbations approximating the space of distortions resulting from real printing and photography. <details>
  		    <summary>Click to expand</summary>We demonstrates real-time decoding of hyperlinks in photos from in-the-wild videos that contain variation in lighting, shadows, perspective, occlusion and viewing distance. Our prototype system robustly retrieves 56 bit hyperlinks after error correction – sufficient to embed a unique code within every photo on the internet.
	        </details>
		</font>
	     	</p>
		<p align="center"><img width="%100" height="45" src="https://eccv2020.eu/wp-content/uploads/2020/05/eccv-online-logo_A.png"> AIM (ECCV 2020)
			<br>[<b><a href="https://arxiv.org/pdf/1904.05343.pdf" target="_blank">arXiv</a></b> | <a href="https://www.matthewtancik.com/stegastamp" target="_blank"><b>Project Page</b></a> |  <a href="https://www.youtube.com/watch?v=E8OqgNDBGO0" target="_blank"><b>Video</b></a>  |  <a href="https://github.com/tancik/StegaStamp" target="_blank"><b>Code</b></a>]
		</p>
	</td>
</tr>
<tr>
	<td><font size="4">1.</font></td>
	<td><center><a href="https://www.youtube.com/watch?v=3YjkkxgAIKw" target="_blank"><img src="https://robinkips.github.io/CA-GAN/images/full_face_shades.png" alt="CA-GAN: Weakly Supervised Color Aware GAN for Controllable Makeup Transfer" width="1022" height="%100" /></a></center>
	</td>
	<td>
		<p align="center"><font size="4"><b>CA-GAN</b>: Weakly Supervised Color Aware GAN for Controllable Makeup Transfer</font>
			<br><b>Robin Kips<sup>*</sup></b>, Pietro Gori, Matthieu Perrot, and Isabelle Bloch
		</p>
		<p align="left">
	        <font size =2>
	        <b>Abstract</b>. While existing makeup style transfer models perform an image synthesis whose results cannot be explicitly controlled, the ability to modify makeup color continuously is a desirable property for virtual try-on applications. We propose a new formulation for the makeup style transfer task, with the objective to learn a color controllable makeup style synthesis. We introduce CA-GAN, a generative model that learns to modify the color of specific objects (e.g. lips or eyes) in the image to an arbitrary target color while preserving background. Since color labels are rare and costly to acquire, <details>
  		    <summary>Click to expand</summary>our method leverages weakly supervised learning for conditional GANs. This enables to learn a controllable synthesis of complex objects, and only requires a weak proxy of the image attribute that we desire to modify. Finally, we present for the first time a quantitative analysis of makeup style transfer and color control performance.
	        <br><b>Keywords</b>: Image Synthesis, GANs, Weakly Supervised Learning, Makeup Style Transfer
	        </details>
		</font>
      	</p>
		<p align="center"><img width="%100" height="30" src="https://cvpr2019.thecvf.com/images/sponsors/cvf_.jpg"> CVPR 2020
			<br>[<b><a href="https://arxiv.org/pdf/2008.10298.pdf" target="_blank">PDF</a></b> | <a href="https://robinkips.github.io/CA-GAN/" target="_blank"><b>Project Page</b></a> |  <a href="https://www.youtube.com/watch?v=3YjkkxgAIKw&feature=emb_logo" target="_blank"><b>Video</b></a>  |  <a href="https://github.com/marsyy/littl_tools/tree/master/bluetooth" target="_blank"><b>Code</b></a>]
		</p>
	</td>
</tr>
</table>
<br>
