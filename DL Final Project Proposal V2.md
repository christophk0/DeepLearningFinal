**Team Name**  
QCTM: Quantize • Compute • Train • Model

**Project Title**  
Comparing vision transformers to convolutional neural networks

**Project summary (4-5+ sentences). Fill in your problem and background/motivation (why do you want to solve it? Why is it interesting?). This should provide some detail (don’t just say “I’ll be working on object detection”)**

Deep learning models for computer vision have evolved from convolutional architectures (CNNs) to attention-based architectures such as Vision Transformers (ViTs). CNNs introduce strong inductive biases like locality and translation invariance, whereas ViTs rely on large-scale data and self-attention to learn such priors implicitly.  
Our central research question is:  
***Under what data conditions do Vision Transformers outperform or underperform CNNs, and what architectural properties explain these behaviors?***  
Specifically, we aim to explore the following questions:

* How performance and generalization differ across data regimes (small datasets, imbalanced classes, corrupted images).  
* How modifying attention locality in ViTs affects their ability to model spatial dependencies.  
* Whether pretraining and fine-tuning mitigate ViTs’ data inefficiency.

**What you will do (Approach, 4-5+ sentences) \- Be specific about what you will implement and what existing code you will use. Describe what you actually plan to implement or the experiments you might try, etc. Again, provide sufficient information describing exactly what you’ll do. One of the key things to note is that just downloading code and running it on a dataset is not sufficient for a description or a project\! Some thorough implementation, analysis, theory, etc. have to be done for the project.**

We will compare the Pytorch implementation of CNNs and vision transformers across several dimensions. Some ideas:

* Performance across different data sets for default implementation  
  * Standard data set  
  * Sparse data set  
  * Many classes  
  * Imbalanced data set  
* Performance when both networks are shallow.  
* Performance when the visual transformer attention mechanism is modified to only allow attention on nearby pixels.  
* Number of weights, training time required to reach certain accuracy thresholds.  
* Performance when using pre-trained networks with fine tuning.  
* Performance when fine tuning pretrained networks with later layers removed.

To implement, we plan to:

* Use PyTorch’s official torchvision.models implementations for ResNet and ViT.  
* Modify ViT’s attention mask to implement local attention.  
* Evaluate metrics including accuracy, F1-score, per-class recall, epochs to convergence, training curve.

**Resources / Related Work & Papers (4-5+ sentences). What is the state of art for this problem? Note that it is perfectly fine for this project to implement approaches that already exist. This part should show you’ve done some research about what approaches exist.**

* [Attention is all you need](https://arxiv.org/abs/1706.03762):   
  * This introduced the Transformer architecture for NLP tasks, replacing recurrence and convolution with self-attention mechanisms.  
* [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
  * Pytorch seems to have support for this [already](https://docs.pytorch.org/vision/main/models/vision_transformer.html)  
  * The finding is that when trained on very large datasets like JFT-300M, ViTs achieve or exceed CNN performance, however, ViTs perform significantly worse on small datasets.  
* [Do Vision Transformers See Like Convolutional Neural Networks?](https://arxiv.org/abs/2108.08810)  
  * The state of art insight is ViTs and CNNs learn different kinds of features; ViTs’ attention leads to more holistic global perception but lower data efficiency

**Datasets (Provide a link to the dataset). This is crucial\! Deep learning is data-driven, so what datasets you use is crucial. One of the key things is to make sure you don’t try to create and especially annotate your own data\! Otherwise, the project will be taken over by this.**

* Smaller datasets: CIFAR10/100 and Imagenet  
* [Coyo-labeled-300m](https://huggingface.co/datasets/kakaobrain/coyo-labeled-300m): “COYO-Labeled-300M is a ImageNet-like dataset. Instead of human labeled 1.25 million samples, it's machine-labeled 300 million samples. This dataset is similar to **JFT-300M** which is not released to the public.”  
* Larger dataset: [LAION-400-MILLION OPEN DATASET](https://laion.ai/blog/laion-400-open-dataset/). The image-text-pairs have been extracted from the Common Crawl web data dump and are from random web pages crawled between 2014 and 2021\. We will propose a heuristic to match images to classes.

**List your Group members.**  
Christoph Kinzel ([ckinzel6@gatech.edu](mailto:ckinzel6@gatech.edu))  
Joyce Gu ([joyce\_gu@gatech.edu](mailto:joyce_gu@gatech.edu))  
Tomás Valdivia Hennig ([thennig3@gatech.edu](mailto:thennig3@gatech.edu))  
Mark Gardner ([mgardner60@gatech.edu](mailto:mgardner60@gatech.edu))

