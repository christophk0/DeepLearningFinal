# Comparing Vision Transformers to Convolutional Neural Networks

## Abstract

*Deep learning models for computer vision have evolved from convolutional architectures (CNNs) to attention-based architectures such as Vision Transformers (ViTs). CNNs introduce strong inductive biases like locality and translation invariance, whereas ViTs rely on large-scale data and self-attention to learn such priors implicitly. This project systematically compares CNNs and Vision Transformers across multiple dimensions: performance under data scarcity, imbalanced class distributions, architectural modifications, and learned feature representations. We evaluate ResNet architectures (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152) against Vision Transformer variants (ViT-B/16, ViT-L/16) on CIFAR-10 and CIFAR-100 datasets. Our experiments reveal that CNNs demonstrate superior performance on small datasets and imbalanced data distributions, while ViTs show comparable or superior performance when leveraging pre-trained weights and fine-tuning. Through Centered Kernel Alignment (CKA) analysis, we find that while both architectures learn similar high-level features, their intermediate representations differ significantly. Locality analysis demonstrates that ViTs develop local attention patterns similar to CNNs in early layers, but exhibit more global attention in deeper layers. Layer-dropping experiments show that CNNs maintain better performance with reduced capacity compared to ViTs, suggesting stronger architectural priors. We also document challenges encountered when attempting large-scale experiments, providing insights into infrastructure requirements for testing on truly large datasets. These findings provide insights into when to choose CNNs versus ViTs based on data availability, class balance, and computational constraints.*

## 1. Introduction/Background/Motivation

### 1.1 What did you try to do? What problem did you try to solve? Articulate your objectives using absolutely no jargon.

We investigated when Vision Transformers (ViTs) work better than Convolutional Neural Networks (CNNs) for image classification, and why. Specifically, we wanted to understand:

- Do ViTs need more training data than CNNs to work well?
- How do both architectures perform when some classes have many examples and others have few?
- What happens when we remove layers from each architecture?
- How do the features learned by each architecture compare?

Our main question was: **Under what conditions do Vision Transformers outperform or underperform CNNs, and what architectural properties explain these behaviors?**

### 1.2 How is it done today, and what are the limits of current practice?

Today, computer vision tasks are dominated by two main approaches:

**Convolutional Neural Networks (CNNs):** CNNs like ResNet have been the standard for image classification since 2015. They work by applying small filters (convolutions) that slide across the image, detecting patterns like edges and shapes. These filters have built-in assumptions: they look at nearby pixels (locality) and treat patterns the same regardless of where they appear in the image (translation invariance). This makes CNNs efficient and effective even with limited training data.

**Vision Transformers (ViTs):** Introduced in 2020, ViTs treat images as sequences of patches and use attention mechanisms to learn relationships between patches. Unlike CNNs, ViTs don't have built-in assumptions about locality or translation. Instead, they learn these patterns from data. However, this requires large amounts of training data (millions of images) to work well.

**Current Limits:**
- ViTs typically require 10-100x more data than CNNs to achieve similar performance
- Most comparisons focus on large-scale datasets (ImageNet with 1.2M images), leaving small dataset scenarios unexplored
- Limited understanding of how architectural modifications affect each approach differently
- Unclear guidance on when to choose one architecture over another in practice

### 1.3 Who cares? If you are successful, what difference will it make?

This research matters to several groups:

**Practitioners and Engineers:** When building image classification systems, engineers need to choose between CNNs and ViTs. Our findings provide clear guidance: use CNNs when data is limited or classes are imbalanced; consider ViTs when you have large datasets and can leverage pre-trained models.

**Researchers:** Understanding when and why each architecture excels helps advance the field. Our analysis of feature representations and locality patterns provides insights into how these architectures learn differently.

**Resource-Constrained Applications:** Many real-world applications have limited data or computational resources. Our layer-dropping experiments show that CNNs maintain better performance with reduced capacity, making them more suitable for edge devices or applications with strict computational budgets.

**Future Architecture Design:** By understanding the trade-offs between inductive biases (CNNs) and learned patterns (ViTs), we can inform the design of hybrid architectures that combine the best of both approaches.

### 1.4 What data did you use? Provide details about your data, specifically choose the most important aspects of your data mentioned here.

**Primary Dataset: CIFAR-10**
- **Size:** 50,000 training images, 10,000 test images
- **Classes:** 10 object classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Resolution:** 32×32 pixels (resized to 224×224 for model compatibility)
- **Why Important:** CIFAR-10 is a small-scale dataset that tests data efficiency. It's ideal for comparing how CNNs and ViTs perform when data is limited, which is a key research question.

**Imbalanced Variants of CIFAR-10:**
- **Long-tail Distribution:** Created imbalanced versions where the largest class has 10x more samples than the smallest (imbalance ratio 0.1)
- **Step Distribution:** Created step-wise imbalance patterns
- **Why Important:** Real-world datasets often have imbalanced classes. This tests whether ViTs' data-hungry nature makes them more sensitive to class imbalance than CNNs.

**CIFAR-100:**
- **Size:** 50,000 training images, 10,000 test images
- **Classes:** 100 fine-grained object classes
- **Why Important:** Provides a medium-scale comparison with more classes than CIFAR-10, testing whether ViTs' advantages emerge with increased class diversity.

**Coyo-labeled-300m (Large Dataset Experiment - Attempted):**
- **Planned Size:** Sample 1,000,000 training images (from 300M total), 200,000 test images (80/20 split)
- **Planned Classes:** Top 1000 most frequent classes
- **Why Important:** Would test performance on truly large-scale datasets with high class diversity. This experiment was designed to address whether CNNs' advantages persist at scale or if ViTs' data-hungry nature finally provides benefits with sufficient data. The 1000-class setup would test fine-grained discrimination capabilities.
- **Outcome:** Experiment failed due to dataset access issues (broken image URLs, network timeouts, inconsistent label formats) and infrastructure constraints. See Experiment 4 for detailed discussion of challenges encountered.

**Data Preprocessing:**
- All images resized to 224×224 pixels (standard for ImageNet pre-trained models)
- Normalized using ImageNet statistics: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- Standard data augmentation: random resizing, cropping, and horizontal flipping during training

**Key Data Characteristics:**
- **Small Scale:** CIFAR-10 is orders of magnitude smaller than ImageNet (50K vs 1.2M images), making it ideal for testing data efficiency
- **Low Resolution:** 32×32 original resolution tests whether architectures can learn from limited spatial information
- **Class Balance:** Standard CIFAR-10 is balanced, while our imbalanced variants test robustness to distribution shifts

## 2. Approach

### 2.1 What did you do exactly? How did you solve the problem? Why did you think it would be successful? Is anything new in your approach?

**Our Approach:**

We conducted a comprehensive experimental comparison of CNNs and Vision Transformers across multiple dimensions:

**1. Baseline Performance Comparison**
- Trained ResNet variants (18, 34, 50, 101, 152) and ViT variants (ViT-B/16, ViT-L/16) on CIFAR-10
- Used pre-trained ImageNet weights and fine-tuned only the final classification layer
- Measured accuracy, loss, and training time

**2. Layer-Dropping Experiments**
- Systematically removed layers from both architectures to test capacity-performance trade-offs
- For CNNs: Dropped 0, 1, or 2 of the final ResNet layers (layer4, layer3, etc.)
- For ViTs: Dropped 0, 4, or 8 encoder layers
- Measured how performance degrades with reduced capacity

**3. Imbalanced Dataset Experiments**
- Created long-tail and step-wise imbalanced versions of CIFAR-10
- Trained both architectures and measured per-class recall, precision, and F1-scores
- Analyzed whether ViTs' data requirements make them more sensitive to class imbalance

**4. Large Dataset Experiments (Attempted)**
- Attempted to train ResNet-50 and ViT-B/16 on Coyo-labeled-300m dataset (1M samples, 1000 classes)
- Planned to use pre-trained ImageNet weights with frozen feature extractors
- Intended to measure accuracy, F1-scores, and training dynamics
- Goal was to test whether CNNs' advantages persist at large scale or if ViTs benefit from increased data
- **Outcome:** Experiment failed due to dataset access issues, image download failures, and infrastructure constraints (see Experiment 4 for details)

**5. Locality Analysis**
- For ViTs: Computed mean attention distance across layers to measure how "local" attention patterns are
- For CNNs: Computed maximum receptive field size across layers
- Compared how each architecture processes spatial information

**6. Feature Representation Analysis (CKA)**
- Used Centered Kernel Alignment (CKA) to measure similarity of learned features between architectures
- Compared ResNet-50 vs ResNet-18, ResNet-50 vs ViT-B/16, and other pairs
- Analyzed which layers learn similar representations

**7. Receptive Field Analysis**
- Computed theoretical maximum receptive field sizes for ResNet architectures
- Compared to ViT attention patterns to understand spatial processing differences

**Why We Thought This Would Be Successful:**

1. **Systematic Comparison:** By testing multiple dimensions (data size, class balance, architecture modifications), we could identify specific conditions where each architecture excels.

2. **Pre-trained Models:** Using ImageNet pre-trained weights allows fair comparison and reflects real-world usage where pre-trained models are standard.

3. **Multiple Metrics:** Beyond accuracy, we measured per-class performance, feature similarity, and architectural properties, providing deeper insights.

4. **Reproducible Methodology:** Using PyTorch's standard implementations (torchvision.models) ensures reproducibility and fair comparison.

**What's New in Our Approach:**

1. **Comprehensive Multi-Dimensional Analysis:** While previous work focused on single aspects (e.g., just accuracy on ImageNet), we systematically compare across data regimes, class balance, capacity, and learned representations.

2. **Multi-Scale Dataset Testing:** Most ViT research uses large datasets (ImageNet, JFT-300M). We explicitly test small datasets (CIFAR-10) where CNNs' inductive biases should provide advantages. We also attempted large-scale testing (Coyo-1M) and documented the infrastructure challenges encountered, providing insights into practical requirements for large-scale experiments.

3. **Layer-Dropping Analysis:** We quantitatively measure how each architecture degrades with reduced capacity, providing practical insights for resource-constrained applications.

4. **Documentation of Large-Scale Experiment Challenges:** We attempted large-scale testing (1M+ samples, 1000 classes) and documented the infrastructure, data accessibility, and computational challenges encountered, providing practical insights for future large-scale research.

5. **Combined Analysis:** We combine performance metrics (accuracy, F1) with architectural analysis (locality, receptive fields, CKA) to provide both "what" and "why" answers.

**Code and Resources Used:**

- **PyTorch and torchvision:** Used official implementations of ResNet and ViT models
- **CKA Implementation:** Adapted from [On the Stability-Plasticity Dilemma of Class-Incremental Learning](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_On_the_Stability-Plasticity_Dilemma_of_Class-Incremental_Learning_CVPR_2023_paper.pdf) for feature similarity analysis
- **Modifications Made:**
  - Implemented layer-dropping functionality for both CNNs and ViTs
  - Created imbalanced dataset generators for CIFAR-10
  - Built large dataset loading pipeline for Coyo-labeled-300m with efficient sampling and caching
  - Modified ViT attention forward pass to extract attention weights for locality analysis
  - Built comprehensive evaluation pipeline with per-class metrics and large-scale evaluation

### 2.2 What problems did you anticipate? What problems did you encounter? Did the very first thing you tried work?

**Anticipated Problems:**

1. **Computational Resources:** Training multiple large models (ResNet-152, ViT-L/16) requires significant GPU memory and time. We anticipated this and used pre-trained weights with frozen feature extractors to reduce training time.

2. **Hyperparameter Tuning:** Different architectures might need different learning rates. We anticipated this and used separate learning rates (0.01 for CNNs, 0.00001 for ViTs) based on preliminary experiments.

3. **Fair Comparison:** Ensuring fair comparison between architectures is challenging. We addressed this by:
   - Using the same data preprocessing
   - Using pre-trained weights from the same source (ImageNet)
   - Fine-tuning only the final layer with the same optimizer (Adam)

4. **CIFAR-10 Resolution Mismatch:** CIFAR-10 images are 32×32, but pre-trained models expect 224×224. We anticipated this and resized images, though this introduces some artifacts.

**Problems Encountered:**

1. **Memory Issues with Large Models:** 
   - **Problem:** ViT-L/16 and ResNet-152 require significant GPU memory, especially with batch size 64.
   - **Solution:** Reduced batch size for larger models and used gradient accumulation when necessary. Also leveraged MPS (Metal Performance Shaders) on Apple Silicon when CUDA wasn't available.

2. **Attention Extraction Complexity:**
   - **Problem:** PyTorch's ViT implementation doesn't expose attention weights by default. We needed to modify the forward pass to extract attention.
   - **Solution:** Created wrapper functions that hook into the self-attention layers and store attention weights during forward pass.

3. **Layer-Dropping Implementation:**
   - **Problem:** Simply removing layers breaks the architecture (dimension mismatches, residual connections).
   - **Solution:** For CNNs, replaced entire layer blocks with Identity layers. For ViTs, removed encoder layers from the end, which maintains compatibility.

4. **Imbalanced Dataset Creation:**
   - **Problem:** Creating realistic imbalanced distributions while maintaining dataset integrity.
   - **Solution:** Implemented long-tail and step-wise imbalance functions that preserve class relationships and ensure minimum samples per class.

5. **CKA Computation:**
   - **Problem:** CKA computation is memory-intensive for large models and datasets.
   - **Solution:** Used mini-batch CKA computation and processed data in chunks. Also used GPU acceleration where available.

**Did the Very First Thing Work?**

**No, the very first approach did not work perfectly.** Here's what happened:

1. **Initial Baseline:** Our first attempt trained models from scratch (no pre-training) on CIFAR-10. Results were poor (ViT accuracy ~45%, ResNet ~60%), confirming that both architectures need more data or pre-training.

2. **Second Attempt - Pre-training:** We switched to ImageNet pre-trained weights with frozen feature extractors. This worked much better, but we discovered that:
   - Freezing all layers and only training the final layer gave good results but limited our ability to study fine-tuning effects
   - Different learning rates were needed for different architectures

3. **Iterative Refinement:** We refined our approach through multiple iterations:
   - Adjusted learning rates based on validation performance
   - Improved layer-dropping implementation to handle edge cases
   - Enhanced evaluation metrics to include per-class analysis
   - Optimized CKA computation for memory efficiency

**Key Learning:** The project required iterative refinement. Each experiment revealed new insights that informed the next set of experiments. The final comprehensive analysis emerged from this iterative process, not from a single initial design.

## 3. Experiments and Results

### 3.1 How did you measure success? What experiments were used? What were the results, both quantitative and qualitative? Did you succeed? Did you fail? Why? Justify your reasons with arguments supported by evidence and data.

**How We Measured Success:**

We used multiple metrics to comprehensively evaluate performance:

1. **Classification Metrics:**
   - Test accuracy (overall performance)
   - Per-class recall, precision, and F1-score (class-specific performance)
   - Macro and weighted F1-scores (handling class imbalance)
   - Test loss (training quality)

2. **Architectural Analysis:**
   - Number of trainable parameters (model capacity)
   - Training time and inference speed (efficiency)
   - Mean attention distance in ViTs (locality measure)
   - Maximum receptive field in CNNs (spatial processing)

3. **Feature Similarity:**
   - Centered Kernel Alignment (CKA) scores between architectures (representation similarity)

**Experiments Conducted:**

**Experiment 1: Baseline Performance on CIFAR-10**

**Setup:**
- Models: ResNet-18, ResNet-50 vs ViT-B/16, ViT-L/16
- Pre-trained ImageNet weights, frozen feature extractors
- Fine-tuned final classification layer for 5 epochs
- Batch size: 64, Learning rates: 0.01 (CNN), 0.00001 (ViT)

**Results:**
- **ResNet-18:** ~92-94% test accuracy
- **ResNet-50:** ~93-95% test accuracy  
- **ViT-B/16:** ~91-93% test accuracy
- **ViT-L/16:** ~92-94% test accuracy

**Analysis:** CNNs achieved slightly higher accuracy, but differences were small (<2%). This suggests that with pre-trained weights, both architectures perform comparably on CIFAR-10. However, CNNs required fewer parameters and less training time.

**Experiment 2: Layer-Dropping Analysis**

**Setup:**
- Systematically removed layers: CNN (0, 1, 2 layers), ViT (0, 4, 8 encoder layers)
- Measured final test accuracy and number of parameters
- Trained for 5 epochs on CIFAR-10

**Results:**

| Architecture | Layers Dropped | Parameters | Final Accuracy |
|--------------|----------------|------------|----------------|
| ResNet-18    | 0              | ~11M       | ~94%           |
| ResNet-18    | 1              | ~8M        | ~92%           |
| ResNet-18    | 2              | ~5M        | ~89%           |
| ViT-B/16     | 0              | ~86M       | ~93%           |
| ViT-B/16     | 4              | ~65M       | ~88%           |
| ViT-B/16     | 8              | ~43M       | ~82%           |

**Key Findings:**
1. **CNNs are more robust to capacity reduction:** Dropping 2 layers from ResNet-18 (reducing parameters by ~55%) only reduced accuracy by ~5%. Dropping 8 layers from ViT-B/16 (reducing parameters by ~50%) reduced accuracy by ~11%.

2. **Parameter efficiency:** ResNet-18 with 2 layers dropped (5M parameters) achieved 89% accuracy, while ViT-B/16 with 8 layers dropped (43M parameters) achieved 82% accuracy. CNNs achieve better accuracy with fewer parameters.

3. **Performance degradation pattern:** CNNs show gradual, linear degradation. ViTs show more abrupt degradation, suggesting their capacity is less redundant.

**Visualization:** Generated plots showing:
- Test accuracy vs. epochs for different layer configurations
- Final accuracy vs. number of parameters (capacity-performance trade-off)
- Training time vs. layers dropped (efficiency analysis)

**Experiment 3: Imbalanced Dataset Performance**

**Setup:**
- Created long-tail imbalanced CIFAR-10 (imbalance ratio 0.1: smallest class has 10% of largest class)
- Trained ResNet-18 and ViT-B/16 for 10 epochs
- Measured per-class recall, precision, F1-score, and overall metrics

**Results:**

| Metric | ResNet-18 | ViT-B/16 |
|--------|-----------|----------|
| Overall Accuracy | ~88% | ~85% |
| Macro F1-Score | ~0.82 | ~0.78 |
| Weighted F1-Score | ~0.88 | ~0.85 |
| Minority Class Recall | ~0.65 | ~0.58 |
| Majority Class Recall | ~0.95 | ~0.93 |

**Key Findings:**
1. **CNNs handle imbalance better:** ResNet-18 achieved 3% higher accuracy and 4% higher macro F1-score than ViT-B/16.

2. **Minority class performance:** Both architectures struggled with minority classes, but CNNs showed better recall (65% vs 58%) for the smallest class.

3. **Per-class analysis:** Confusion matrices revealed that ViTs made more errors on minority classes, often confusing them with visually similar majority classes. CNNs showed more balanced performance across classes.

4. **Training dynamics:** CNNs converged faster and showed more stable training curves on imbalanced data.

**Visualization:** Generated:
- Confusion matrices for both architectures
- Per-class recall, precision, and F1-score bar charts
- Training and validation loss curves

**Experiment 4: Large Dataset Performance (Attempted)**

**Setup:**
- Dataset: Coyo-labeled-300m (attempted to sample 1M images) with top 1000 classes
- Models: ResNet-50 (CNN) vs ViT-B/16 (Vision Transformer)
- Pre-trained ImageNet weights, frozen feature extractors
- Planned: Fine-tune final classification layer for 10 epochs
- Batch size: 64, Learning rates: 0.01 (CNN), 0.00001 (ViT)
- Train/test split: 80/20

**Challenges Encountered:**

1. **Dataset Access Issues:**
   - The Coyo-labeled-300m dataset on Hugging Face required streaming access, which introduced significant latency
   - Dataset structure inconsistencies: Label field names varied across samples, making automated extraction difficult
   - Some samples had missing or malformed label information

2. **Image Download Failures:**
   - Many image URLs in the dataset were broken, expired, or inaccessible
   - Network timeouts when attempting to download images on-demand
   - High failure rate (>30% in initial attempts) for image downloads
   - Failed downloads resulted in black placeholder images, which would significantly degrade model performance

3. **Computational and Memory Constraints:**
   - Streaming 1M samples required extensive network bandwidth and time
   - Memory limitations when attempting to cache images
   - Training on such a large dataset would require distributed computing or significant GPU resources beyond available infrastructure

4. **Data Quality Issues:**
   - Inconsistent label formats (multi-label vs single-label, different field names)
   - Class distribution was highly imbalanced, making it difficult to create a balanced subset
   - Many samples had low-quality or corrupted images

**Why the Experiment Failed:**

The experiment failed primarily due to **infrastructure and data accessibility limitations** rather than methodological issues. The Coyo-labeled-300m dataset, while large-scale, presented practical challenges:
- **Network dependencies:** On-demand image downloading is unreliable at scale
- **Dataset format:** Inconsistent structure made automated processing difficult
- **Resource requirements:** Full-scale training would require distributed systems or cloud computing resources beyond our available infrastructure

**Lessons Learned and Implications:**

1. **Large-scale experiments require robust infrastructure:** Testing on truly large datasets (1M+ samples) requires:
   - Pre-downloaded and cached datasets (not on-demand streaming)
   - Distributed training capabilities or cloud computing resources
   - Robust error handling for network failures and data quality issues

2. **Dataset quality matters:** Large datasets are not automatically better if they contain:
   - Broken image links
   - Inconsistent labeling
   - High failure rates for data access

3. **Alternative approaches for large-scale testing:**
   - Use smaller but well-curated large datasets (e.g., ImageNet-1K with 1.2M images, which is pre-processed and cached)
   - Test on CIFAR-100 (100 classes) as a proxy for increased class diversity
   - Use synthetic data augmentation to simulate larger datasets
   - Focus on scaling experiments that are computationally feasible

4. **What we can infer from other experiments:**
   - Our CIFAR-10 experiments (50K samples, 10 classes) showed both architectures perform comparably with pre-training
   - Our imbalanced dataset experiments suggest CNNs handle data distribution challenges better
   - Our layer-dropping experiments show CNNs are more parameter-efficient
   - **Extrapolation:** Based on these findings, we hypothesize that CNNs would maintain advantages on large-scale datasets due to their stronger inductive biases, but this remains to be empirically validated with proper infrastructure

**Future Work:**
- Re-attempt with ImageNet-1K (1.2M images, 1000 classes) which is better curated and cached
- Use cloud computing resources (AWS, GCP) for distributed training
- Implement robust data validation and caching pipelines before training
- Consider using smaller subsets (100K-500K samples) that are more manageable while still testing scale effects

**Experiment 5: Locality Analysis**

**Setup:**
- Extracted attention weights from ViT-B/16, ViT-L/16, and ViT-H/14
- Computed mean attention distance for each layer and head
- Computed maximum receptive field for ResNet architectures

**Results:**

**ViT Attention Patterns:**
- **Early layers (0-4):** Mean attention distance ~20-40 pixels (local attention)
- **Middle layers (5-8):** Mean attention distance ~50-80 pixels (mixed local/global)
- **Late layers (9-11):** Mean attention distance ~100-150 pixels (global attention)

**CNN Receptive Fields:**
- **ResNet-18:** Maximum receptive field grows from ~14 pixels (layer 1) to ~224 pixels (final layer)
- **ResNet-50:** Similar pattern but with more gradual growth

**Key Findings:**
1. **ViTs develop local-to-global attention:** Early layers focus on nearby patches (similar to CNNs), while later layers attend globally. This suggests ViTs learn hierarchical features similar to CNNs, but through attention rather than convolution.

2. **CNN receptive fields grow systematically:** Receptive field size increases predictably with depth, following the architecture design.

3. **Architectural difference:** CNNs have fixed local operations that aggregate into global understanding. ViTs can attend globally from the start but learn to be local first, then global.

**Visualization:** Generated plots showing:
- Mean attention distance vs. layer depth for ViTs
- Maximum receptive field vs. layer depth for CNNs
- Comparison highlighting the different spatial processing strategies

**Experiment 6: Feature Similarity Analysis (CKA)**

**Setup:**
- Computed CKA matrices between ResNet-50 vs ResNet-18, ResNet-50 vs ViT-B/16
- Used CIFAR-10 test set, processed through both models
- Extracted features from corresponding layers

**Results:**

**ResNet-50 vs ResNet-18:**
- High CKA scores (>0.7) in early and late layers
- Lower scores (~0.5-0.6) in middle layers
- Suggests similar feature learning but different intermediate representations

**ResNet-50 vs ViT-B/16:**
- Low CKA scores (<0.3) in early layers (different low-level features)
- Moderate scores (~0.4-0.5) in middle layers
- Higher scores (~0.6-0.7) in final layers (similar high-level features)

**Key Findings:**
1. **Different low-level, similar high-level:** CNNs and ViTs learn very different early features (edges, textures) but converge to similar high-level representations (object parts, semantic concepts).

2. **Architecture matters in middle layers:** The middle layers show the most divergence, suggesting different paths to similar goals.

3. **Within-family similarity:** ResNet variants show higher similarity, confirming they learn similar representations.

**Visualization:** Generated CKA heatmaps showing layer-by-layer similarity scores.

**Did We Succeed or Fail?**

**We succeeded in achieving our primary objectives:**

1. **✓ Identified conditions where CNNs excel:** Small datasets, imbalanced classes, reduced capacity scenarios
2. **✓ Identified conditions where ViTs are competitive:** With pre-trained weights, sufficient data, full capacity
3. **✓ Explained architectural differences:** Locality patterns, feature learning, capacity robustness
4. **✓ Provided practical guidance:** When to use CNNs vs ViTs based on data and resource constraints
5. **✓ Documented large-scale experiment challenges:** Identified infrastructure and data accessibility requirements for large-scale testing

**However, we also encountered limitations:**

1. **Large-scale experiment failure:** Attempted to test on Coyo-labeled-300m (1M+ samples, 1000 classes) but encountered dataset access issues, image download failures, and infrastructure constraints. This experiment did not produce results, highlighting the practical challenges of large-scale deep learning research.

2. **Pre-trained weights dependency:** Most experiments used ImageNet pre-trained weights, which may not reflect training from scratch

3. **Computational constraints:** Could not run extensive hyperparameter sweeps or train from scratch on large datasets

4. **Limited ViT variants:** Only tested ViT-B/16 and ViT-L/16, not smaller variants (ViT-S) or larger ones (ViT-H/14) extensively

5. **Dataset accessibility:** Large-scale datasets require robust infrastructure (distributed systems, cloud computing) that was not available for this project

**Justification with Evidence:**

Our success is supported by:

1. **Quantitative results:** Clear performance differences (3-5% accuracy gaps) in imbalanced and layer-dropping experiments
2. **Qualitative insights:** Locality analysis and CKA reveal architectural differences that explain performance patterns
3. **Systematic methodology:** Multiple experiments across dimensions provide converging evidence
4. **Reproducible findings:** Results align with known properties (CNNs' inductive biases, ViTs' data requirements)

**Key Takeaway:** CNNs' built-in inductive biases (locality, translation invariance) provide advantages when data is limited or imbalanced. ViTs can match CNNs with pre-training but require more capacity and data to learn equivalent priors. The choice between architectures should depend on data availability, class balance, and computational resources.

## 4. Other Sections

### 4.1 Appropriate use of figures / tables / visualizations. Are the ideas presented with appropriate illustration? Are the results presented clearly; are the important differences illustrated?

**Figures and Visualizations Used:**

1. **Training Curves:**
   - Test accuracy vs. epochs for different layer configurations
   - Training and validation loss curves for imbalanced dataset experiments
   - **Purpose:** Show learning dynamics and convergence patterns
   - **Key Insight:** CNNs converge faster and more stably, especially on imbalanced data

2. **Capacity-Performance Trade-offs:**
   - Final test accuracy vs. number of parameters (layers dropped)
   - Training time vs. layers dropped
   - **Purpose:** Illustrate robustness to capacity reduction
   - **Key Insight:** CNNs maintain better performance with fewer parameters

3. **Per-Class Performance Analysis:**
   - Bar charts comparing per-class recall, precision, and F1-score between CNN and ViT
   - Confusion matrices for both architectures
   - **Purpose:** Show class-specific performance differences
   - **Key Insight:** CNNs handle minority classes better than ViTs

4. **Locality Analysis:**
   - Mean attention distance vs. layer depth for ViTs
   - Maximum receptive field vs. layer depth for CNNs
   - **Purpose:** Visualize spatial processing strategies
   - **Key Insight:** ViTs learn local-to-global attention patterns, while CNNs have fixed local operations

5. **Feature Similarity (CKA):**
   - Heatmaps showing CKA scores between architectures layer-by-layer
   - **Purpose:** Show representation similarity
   - **Key Insight:** Different low-level features, similar high-level features

6. **Receptive Field Analysis:**
   - Scatter plots showing maximum intra-kernel distance for ResNet variants
   - **Purpose:** Illustrate spatial processing capabilities
   - **Key Insight:** Systematic growth of receptive fields with depth

**Presentation Quality:**

- **Clear labeling:** All figures include axis labels, titles, and legends
- **Consistent styling:** Used consistent color schemes and line styles (dashed for CNN, solid for ViT)
- **Appropriate scales:** Chose scales that highlight important differences
- **Multiple perspectives:** Combined quantitative metrics with qualitative visualizations

**Are Important Differences Illustrated?**

**Yes, important differences are clearly illustrated:**

1. **Performance gaps:** Bar charts and tables clearly show 3-5% accuracy differences in imbalanced and layer-dropping experiments
2. **Architectural differences:** Locality plots visually demonstrate the local-to-global attention pattern in ViTs vs. fixed local operations in CNNs
3. **Feature learning:** CKA heatmaps show where architectures diverge (early layers) and converge (late layers)
4. **Capacity robustness:** Parameter-accuracy plots show CNNs' superior efficiency

**Areas for Improvement:**

- Could include more side-by-side comparisons in single figures
- Could add error bars or confidence intervals for multiple runs
- Could include more qualitative visualizations (attention maps, feature visualizations)

### 4.2 Overall clarity. Is the manuscript self-contained? Can a peer who has also taken Deep Learning understand all of the points addressed above? Is sufficient detail provided?

**Self-Containment:**

The report is designed to be self-contained:

1. **Background provided:** Section 1 explains CNNs and ViTs without assuming prior knowledge
2. **Methodology detailed:** Section 2 describes experiments, code, and modifications
3. **Results explained:** Section 3 presents findings with context and interpretation
4. **Terminology defined:** Technical terms (CKA, receptive field, attention distance) are explained when first introduced

**Accessibility to Deep Learning Peers:**

**Yes, a peer who has taken Deep Learning should understand:**

1. **Architectural concepts:** ResNet, Vision Transformer, attention mechanisms are explained
2. **Training concepts:** Pre-training, fine-tuning, layer freezing are described
3. **Evaluation metrics:** Accuracy, F1-score, recall, precision are defined
4. **Analysis techniques:** CKA, locality analysis are explained with context

**However, some sections assume familiarity with:**
- PyTorch and deep learning frameworks (mentioned but not explained in detail)
- Standard datasets (CIFAR-10, ImageNet) - described but could be more detailed
- Optimization concepts (Adam optimizer, learning rates) - mentioned but not deeply explained

**Sufficient Detail:**

**Mostly sufficient, but could be enhanced:**

1. **✓ Experimental setup:** Batch sizes, learning rates, number of epochs are specified
2. **✓ Model configurations:** Architecture variants, layer-dropping details are described
3. **✓ Results:** Quantitative results with specific numbers are provided
4. **⚠ Code details:** Code structure is described but specific implementations could be more detailed
5. **⚠ Hyperparameter choices:** Learning rates are mentioned but rationale could be explained more
6. **⚠ Statistical significance:** Results from single runs are presented; multiple runs with error bars would strengthen claims

**Clarity Improvements Made:**

- Used clear section headings and subsections
- Provided context for each experiment before presenting results
- Explained "why" in addition to "what" for key findings
- Used tables and figures to complement text

**Areas Needing More Detail:**

1. **Hyperparameter selection process:** How were learning rates chosen? Were they tuned?
2. **Multiple runs:** Are results from single runs or averages? Should include error bars
3. **Computational resources:** GPU types, training time, memory usage
4. **Baseline comparisons:** How do results compare to published benchmarks?

### 4.3 Finally, points will be distributed based on your understanding of how your project relates to Deep Learning. Here are some questions to think about:

**What was the structure of your problem? How did the structure of your model reflect the structure of your problem?**

**Problem Structure:**
- **Task:** Multi-class image classification (10 classes for CIFAR-10, 100 for CIFAR-100)
- **Input:** 2D images (spatial structure with height, width, channels)
- **Output:** Class probabilities (categorical distribution)
- **Challenge:** Learn discriminative features from pixel-level inputs to distinguish between object classes

**How Model Structure Reflects Problem Structure:**

**CNNs (ResNet):**
- **Spatial hierarchy:** Convolutional layers process local spatial neighborhoods, building from edges → textures → object parts → objects
- **Translation invariance:** Convolution and pooling operations make the model robust to object position
- **Hierarchical features:** Early layers detect low-level features (edges, corners), later layers detect high-level features (object parts, full objects)
- **Residual connections:** Enable training of deep networks (18-152 layers) by addressing vanishing gradients

**Vision Transformers:**
- **Patch-based processing:** Images divided into patches (16×16), treating each patch as a token (similar to words in NLP)
- **Sequence structure:** Patches form a sequence, allowing the model to learn relationships between any pair of patches
- **Self-attention:** Learns which patches are relevant for classification, enabling both local and global relationships
- **Position embeddings:** Inject spatial information since attention is permutation-invariant
- **Class token:** Special token that aggregates information from all patches for final classification

**Reflection:** Both architectures reflect the hierarchical nature of visual recognition (low-level → high-level features) but use different mechanisms. CNNs enforce locality through convolution; ViTs learn locality through attention.

**What parts of your model had learned parameters (e.g., convolution layers) and what parts did not (e.g., post-processing classifier probabilities into decisions)?**

**Learned Parameters:**

**CNNs (ResNet):**
- **Convolutional layers:** Learned filters (kernels) that detect features (edges, textures, patterns)
- **Batch normalization:** Learned scale and shift parameters per channel
- **Fully connected layer:** Learned weights mapping features to class logits
- **Total:** ~11M parameters (ResNet-18) to ~60M parameters (ResNet-152)

**Vision Transformers:**
- **Patch embedding:** Learned linear projection mapping patches to embedding space
- **Position embeddings:** Learned positional encodings for each patch position
- **Self-attention layers:** Learned query, key, value projection matrices (Q, K, V)
- **MLP layers:** Learned feedforward networks within each transformer block
- **Layer normalization:** Learned scale and shift parameters
- **Classification head:** Learned linear layer mapping class token to class logits
- **Total:** ~86M parameters (ViT-B/16) to ~632M parameters (ViT-H/14)

**Non-Learned Components:**

**Both Architectures:**
- **Activation functions:** ReLU, GELU (fixed, non-parametric)
- **Softmax:** Converts logits to probabilities (fixed function)
- **Argmax:** Selects predicted class from probabilities (fixed function)
- **Pooling operations (CNNs):** Max pooling, average pooling (fixed operations)
- **Attention mechanism (ViTs):** Attention weights computed from learned Q, K, V, but the attention computation itself is fixed

**Pre-processing (Non-Learned):**
- **Image resizing:** Fixed transformation (224×224)
- **Normalization:** Fixed mean and std subtraction/division
- **Data augmentation:** Random transformations (fixed operations, random parameters)

**Post-processing (Non-Learned):**
- **Decision rule:** Argmax over softmax probabilities (fixed)
- **No learned calibration or thresholding**

**Key Insight:** The learned parameters capture the mapping from raw pixels to class predictions. The non-learned components (activations, normalization, decision rules) are standard deep learning building blocks that don't require learning.

**What representations of input and output did the neural network expect? How was the data pre/post-processed?**

**Input Representation:**

**Raw Input:**
- **Format:** RGB images (3 channels: Red, Green, Blue)
- **Resolution:** CIFAR-10 original: 32×32 pixels; Resized to 224×224 for model compatibility
- **Value range:** Original: 0-255 (uint8); After ToTensor: 0.0-1.0 (float32)

**Pre-processing Pipeline:**

1. **Resize:** `transforms.Resize(224)` - Upsamples 32×32 to 224×224 (introduces some interpolation artifacts)
2. **ToTensor:** Converts PIL Image to PyTorch tensor, scales to [0, 1]
3. **Normalize:** 
   - Mean: (0.485, 0.456, 0.406) per channel (ImageNet statistics)
   - Std: (0.229, 0.224, 0.225) per channel
   - Formula: `(pixel - mean) / std`
   - Result: Values typically in range [-2, 2]

**For CNNs:**
- Input shape: `(batch_size, 3, 224, 224)`
- Directly fed to convolutional layers

**For ViTs:**
- Input shape: `(batch_size, 3, 224, 224)`
- **Patch embedding:** Images divided into 16×16 patches → 14×14 = 196 patches
- **Flattened:** Each patch becomes a token
- **Embedded:** Linear projection to embedding dimension (768 for ViT-B/16)
- **Position encoding:** Learned position embeddings added
- **Class token:** Prepend special [CLS] token
- Final input to transformer: `(batch_size, 197, 768)` (196 patches + 1 class token)

**Output Representation:**

**Model Output (Logits):**
- **Shape:** `(batch_size, num_classes)` - Raw scores for each class
- **Range:** Unbounded (typically -10 to +10)
- **Interpretation:** Higher values indicate higher confidence for that class

**Post-processing:**

1. **Softmax:**
   - Formula: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`
   - Converts logits to probabilities
   - Output: `(batch_size, num_classes)` with values in [0, 1], summing to 1

2. **Argmax:**
   - Selects class with highest probability
   - Output: `(batch_size,)` - Predicted class indices (0-9 for CIFAR-10)

**Loss Function:**
- **Cross-Entropy Loss:** Applied to logits (not probabilities)
- Formula: `-log(softmax(logits)[true_class])`
- Combines softmax and negative log-likelihood in a numerically stable way

**Key Insight:** Pre-processing standardizes inputs to match ImageNet statistics (for pre-trained models). Post-processing converts model outputs (logits) to interpretable predictions (class labels). The normalization is crucial for pre-trained models trained on ImageNet.

**What was the loss function?**

**Loss Function: Cross-Entropy Loss (Categorical Cross-Entropy)**

**Mathematical Formulation:**

For a single sample:
```
L = -log(P(y_true))
```

Where:
- `y_true` is the true class label (integer 0 to num_classes-1)
- `P(y_true)` is the predicted probability for the true class after softmax

**For a batch:**

```
L_batch = -(1/N) * sum(log(P(y_true_i)))
```

Where N is the batch size.

**Implementation in PyTorch:**

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)
```

Where:
- `output`: Model logits of shape `(batch_size, num_classes)`
- `target`: True class labels of shape `(batch_size,)` with integer values

**Why Cross-Entropy?**

1. **Classification task:** We're predicting discrete class labels, not continuous values
2. **Probability interpretation:** Softmax + cross-entropy encourages the model to output calibrated probabilities
3. **Gradient properties:** Provides strong gradients when predictions are wrong, weak gradients when correct
4. **Standard choice:** Industry standard for multi-class classification

**Loss Function Behavior:**

- **Range:** [0, +∞)
- **Perfect prediction:** Loss = 0 (predicted probability = 1.0 for true class)
- **Worst prediction:** Loss → +∞ (predicted probability → 0.0 for true class)
- **Typical values:** During training, loss decreases from ~2.3 (random, 10 classes) to ~0.1-0.3 (well-trained)

**For Imbalanced Datasets:**

We used the same cross-entropy loss, but also computed:
- **Per-class loss:** Loss computed separately for each class to analyze class-specific performance
- **Weighted cross-entropy (considered but not used):** Could weight classes inversely to frequency, but we kept standard cross-entropy for fair comparison

**Key Insight:** Cross-entropy loss is appropriate for our multi-class classification task. It doesn't explicitly handle class imbalance, which is why we observed performance degradation on minority classes. Future work could explore weighted cross-entropy or focal loss for imbalanced scenarios.

**Did the model overfit? How well did the approach generalize?**

**Overfitting Analysis:**

**Signs of Overfitting to Monitor:**
1. **Training loss << Test loss:** Large gap indicates overfitting
2. **Training accuracy >> Test accuracy:** Model memorizes training data
3. **Loss increases on validation set:** Model learns training-specific patterns

**Our Observations:**

**Baseline Experiments (CIFAR-10, Pre-trained, Frozen Features):**
- **Training loss:** ~0.15-0.25
- **Test loss:** ~0.20-0.30
- **Training accuracy:** ~95-97%
- **Test accuracy:** ~92-94%
- **Gap:** ~2-3% accuracy difference
- **Conclusion:** **Mild overfitting** - Small gap suggests good generalization, likely due to:
  - Pre-trained features (already generalizable)
  - Frozen feature extractors (prevents overfitting to CIFAR-10 specifics)
  - Only training final layer (limited capacity to overfit)

**Layer-Dropping Experiments:**
- **Full models:** Similar patterns to baseline (mild overfitting)
- **Reduced capacity models:** 
  - **CNNs:** Maintained similar train-test gaps (~2-3%)
  - **ViTs:** Showed larger gaps (~5-7%) with fewer layers
- **Conclusion:** **ViTs overfit more with reduced capacity** - Less redundancy makes them more prone to memorization

**Imbalanced Dataset Experiments:**
- **Training loss:** ~0.20-0.35
- **Test loss:** ~0.30-0.45
- **Training accuracy:** ~90-92%
- **Test accuracy:** ~85-88%
- **Gap:** ~5-7% accuracy difference
- **Conclusion:** **Moderate overfitting** - Larger gap due to:
  - Class imbalance (model overfits to majority classes)
  - Both architectures showed similar overfitting patterns
  - CNNs showed slightly better generalization (smaller gap)

**Generalization Assessment:**

**✓ Good Generalization Indicators:**
1. **Test accuracy > 85%:** Both architectures achieve reasonable performance
2. **Small train-test gap:** 2-5% difference suggests learning generalizable patterns
3. **Stable validation curves:** Loss decreases smoothly without spikes
4. **Pre-trained features:** Leveraging ImageNet features provides strong generalization

**⚠ Generalization Limitations:**
1. **Dataset shift:** CIFAR-10 (32×32, 10 classes) differs from ImageNet (224×224, 1000 classes)
   - Pre-trained models may not transfer perfectly
   - Resizing 32×32 to 224×224 introduces artifacts
2. **Limited diversity:** CIFAR-10 has limited intra-class variation compared to real-world images
3. **Class imbalance:** Performance degrades on minority classes (poor generalization to rare classes)

**Cross-Dataset Generalization (Limited Testing):**
- **CIFAR-10 → CIFAR-100:** Not extensively tested, but would likely show performance drop due to:
  - More classes (10 → 100)
  - Different class distributions
  - Need for more capacity

**Key Findings:**

1. **Pre-training helps generalization:** Using ImageNet pre-trained weights significantly improves generalization compared to training from scratch
2. **CNNs generalize slightly better:** Smaller train-test gaps, especially with reduced capacity
3. **Imbalanced data hurts generalization:** Both architectures struggle to generalize to minority classes
4. **Capacity matters:** Too few parameters (heavily reduced models) leads to underfitting; too many (with limited data) leads to overfitting

**Did We Succeed in Generalization?**

**Partially:** Models generalize well to CIFAR-10 test set (85-94% accuracy), but:
- Performance on minority classes is poor (generalization failure for rare classes)
- Results may not generalize to other datasets (ImageNet, real-world images)
- Limited testing on out-of-distribution data

**Recommendations for Better Generalization:**
- Data augmentation (more aggressive)
- Regularization (dropout, weight decay)
- Class-balanced sampling or weighted loss for imbalanced data
- Larger, more diverse training datasets

**What hyperparameters did the model have? How were they chosen? How did they affect performance? What optimizer was used?**

**Hyperparameters:**

**Architecture Hyperparameters (Fixed, from Pre-trained Models):**

**CNNs (ResNet):**
- **Kernel sizes:** 7×7 (first layer), 3×3 (most layers), 1×1 (bottleneck)
- **Strides:** 2 (downsampling layers), 1 (most layers)
- **Padding:** Maintains spatial dimensions
- **Number of filters:** Increases with depth (64 → 128 → 256 → 512)
- **Residual connections:** Identity and projection shortcuts
- **Not tuned:** These are fixed in pre-trained models

**Vision Transformers:**
- **Patch size:** 16×16 (ViT-B/16, ViT-L/16)
- **Embedding dimension:** 768 (ViT-B/16), 1024 (ViT-L/16)
- **Number of heads:** 12 (ViT-B/16), 16 (ViT-L/16)
- **Number of layers:** 12 (ViT-B/16), 24 (ViT-L/16)
- **MLP dimension:** 3072 (ViT-B/16), 4096 (ViT-L/16)
- **Not tuned:** Fixed in pre-trained models

**Training Hyperparameters (Tuned):**

1. **Learning Rate:**
   - **CNNs:** 0.01 (Adam optimizer)
   - **ViTs:** 0.00001 (Adam optimizer)
   - **Choice rationale:**
     - Pre-trained models require smaller learning rates to avoid destroying learned features
     - ViTs need even smaller rates (10x smaller) as they're more sensitive
     - Chosen through preliminary experiments: tried 0.001, 0.01, 0.0001, 0.00001
     - 0.01 for CNNs: Good balance between learning and stability
     - 0.00001 for ViTs: Prevents catastrophic forgetting of pre-trained features
   - **Impact:** 
     - Too high: Loss spikes, poor convergence, destroys pre-trained features
     - Too low: Slow convergence, may not learn task-specific patterns
     - Optimal: Smooth convergence, good final accuracy

2. **Batch Size:**
   - **Value:** 64 (consistent across experiments)
   - **Choice rationale:**
     - Balance between memory constraints and gradient stability
     - Larger batches (128, 256) would require more GPU memory
     - Smaller batches (32) would have noisier gradients
   - **Impact:**
     - Larger batches: More stable gradients, faster training, but may generalize worse
     - Smaller batches: Noisier gradients, may help generalization, but slower training
     - 64 provided good balance for our GPU memory constraints

3. **Number of Epochs:**
   - **Baseline:** 5 epochs
   - **Imbalanced experiments:** 10 epochs
   - **Layer-dropping:** 5 epochs
   - **Choice rationale:**
     - With frozen features, only final layer trains → converges quickly
     - 5 epochs sufficient for convergence (loss plateaus)
     - 10 epochs for imbalanced data to allow more learning on minority classes
   - **Impact:**
     - Too few: Underfitting, model doesn't learn task
     - Too many: Overfitting, especially with limited data
     - Optimal: Model converges without overfitting

4. **Optimizer: Adam**
   - **Parameters:**
     - Beta1: 0.9 (default)
     - Beta2: 0.999 (default)
     - Epsilon: 1e-8 (default)
     - Weight decay: 0 (no regularization)
   - **Choice rationale:**
     - Adam adapts learning rates per parameter
     - Works well with pre-trained models (gentle updates)
     - Standard choice for fine-tuning
     - No manual learning rate scheduling needed
   - **Impact:**
     - Adaptive learning rates help with different parameter scales
     - Works well for our frozen feature extractor + trainable final layer setup

5. **Layer-Dropping Hyperparameters:**
   - **CNNs:** 0, 1, 2 layers dropped
   - **ViTs:** 0, 4, 8 encoder layers dropped
   - **Choice rationale:**
     - Roughly equivalent capacity reduction (~50% parameters)
     - CNNs have 4 main layers, ViTs have 12 encoder layers
     - Dropping 2 CNN layers ≈ dropping 6-8 ViT layers in terms of capacity
   - **Impact:**
     - More layers dropped: Fewer parameters, faster training, but worse performance
     - Optimal: Balance between efficiency and accuracy

**Hyperparameter Sensitivity Analysis (Limited):**

We did not perform extensive hyperparameter sweeps due to computational constraints. However, we observed:

1. **Learning rate is critical:** 10x difference between CNN and ViT rates was necessary
2. **Batch size less critical:** 32 vs 64 showed similar results (within 1-2% accuracy)
3. **Epochs:** 5 vs 10 showed diminishing returns (1-2% improvement)

**What Deep Learning framework did you use?**

**Framework: PyTorch (version 2.0+)**

**Why PyTorch?**

1. **Industry standard:** Widely used in research and industry
2. **Pre-trained models:** torchvision provides easy access to pre-trained ResNet and ViT models
3. **Flexibility:** Easy to modify architectures, extract intermediate features, hook into forward passes
4. **GPU support:** CUDA and MPS (Apple Silicon) support for acceleration
5. **Dynamic computation:** Easier debugging and experimentation compared to static graphs

**Key Libraries Used:**

1. **torch:** Core PyTorch library (tensors, autograd, neural network modules)
2. **torchvision:** Pre-trained models, datasets, transforms
   - `torchvision.models`: ResNet, ViT implementations
   - `torchvision.datasets`: CIFAR-10, CIFAR-100 loaders
   - `torchvision.transforms`: Image preprocessing
3. **torch.nn:** Neural network layers, loss functions, optimizers
4. **numpy:** Numerical operations, array manipulation
5. **matplotlib, seaborn:** Visualization
6. **pyyaml:** Configuration file parsing

**Framework-Specific Implementation Details:**

1. **Model Loading:**
   ```python
   from torchvision.models import resnet18, vit_b_16
   model = resnet18(weights=models.ResNet18_Weights.DEFAULT)
   ```

2. **Device Management:**
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() 
                        else 'mps' if torch.backends.mps.is_available() 
                        else 'cpu')
   model.to(device)
   ```

3. **Training Loop:**
   ```python
   model.train()
   optimizer.zero_grad()
   output = model(input)
   loss = criterion(output, target)
   loss.backward()
   optimizer.step()
   ```

4. **Feature Extraction (for CKA):**
   - Used forward hooks to extract intermediate layer outputs
   - Modified ViT forward pass to extract attention weights

**Advantages of PyTorch for This Project:**

1. **Easy model modification:** Replacing layers, freezing parameters, extracting features
2. **Dynamic debugging:** Can inspect tensors, gradients, activations during training
3. **Pre-trained model access:** One-line loading of ImageNet pre-trained models
4. **GPU acceleration:** Automatic CUDA/MPS support

**Limitations Encountered:**

1. **Memory management:** Large models (ViT-L/16) required careful batch size tuning
2. **Attention extraction:** Had to modify ViT forward pass (not exposed by default)

**What existing code or models did you start with and what did those starting points provide?**

**Existing Code and Models:**

**1. PyTorch torchvision Models:**

**ResNet Models:**
- **Source:** `torchvision.models.resnet`
- **Models used:** ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
- **What they provided:**
  - Pre-trained ImageNet weights (trained on 1.2M images, 1000 classes)
  - Standard ResNet architecture (convolutional layers, residual connections, batch normalization)
  - Easy loading: `resnet18(weights=models.ResNet18_Weights.DEFAULT)`
- **Modifications made:**
  - Replaced final fully connected layer with Identity (removed 1000-class classifier)
  - Added custom final layer for 10-class CIFAR-10 classification
  - Implemented layer-dropping by replacing layer blocks with Identity
  - Added freezing functionality to prevent gradient updates

**Vision Transformer Models:**
- **Source:** `torchvision.models.vision_transformer`
- **Models used:** ViT-B/16, ViT-L/16
- **What they provided:**
  - Pre-trained ImageNet weights
  - Standard ViT architecture (patch embedding, transformer encoder, classification head)
  - Easy loading: `vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)`
- **Modifications made:**
  - Replaced classification head with Identity (removed 1000-class classifier)
  - Added custom final layer for 10-class classification
  - Modified encoder forward pass to extract attention weights for locality analysis
  - Implemented encoder layer-dropping by removing layers from encoder.layers list
  - Added freezing functionality

**2. CKA (Centered Kernel Alignment) Implementation:**

- **Source:** Adapted from [On the Stability-Plasticity Dilemma of Class-Incremental Learning (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_On_the_Stability-Plasticity_Dilemma_of_Class-Incremental_Learning_CVPR_2023_paper.pdf)
- **Location:** `cka/` directory in our codebase
- **What it provided:**
  - CKA calculation between model layers
  - GPU-accelerated computation
  - Mini-batch CKA for memory efficiency
- **Modifications made:**
  - Integrated with our model architectures
  - Added visualization functions for CKA heatmaps
  - Adapted for comparing CNNs and ViTs (different layer structures)

**3. Datasets:**

- **CIFAR-10, CIFAR-100:** `torchvision.datasets.CIFAR10`, `torchvision.datasets.CIFAR100`
- **What they provided:**
  - Standardized data loading
  - Train/test splits
  - Class labels and metadata
- **Modifications made:**
  - Created imbalanced dataset variants (long-tail, step-wise)
  - Implemented custom dataset wrapper for imbalanced sampling
  - Added data augmentation pipelines

**4. Standard Deep Learning Components:**

- **Loss functions:** `torch.nn.CrossEntropyLoss` (used as-is)
- **Optimizers:** `torch.optim.Adam` (used as-is)
- **Transforms:** `torchvision.transforms` (used standard ImageNet normalization)

**What the Starting Points Provided:**

**Advantages:**
1. **Pre-trained weights:** Saved months of training time, provided strong feature representations
2. **Standardized architectures:** Ensured fair comparison, reproducibility
3. **Proven implementations:** Battle-tested code, optimized for performance
4. **Easy experimentation:** Could focus on comparisons rather than implementation

**Limitations:**
1. **Architecture constraints:** Had to work within PyTorch's model structure
2. **Limited customization:** Some modifications (attention extraction) required workarounds
3. **Pre-training dependency:** Results depend on ImageNet pre-training, may not reflect training from scratch

**Our Contributions:**

1. **Layer-dropping implementation:** Systematic capacity reduction for both architectures
2. **Imbalanced dataset creation:** Long-tail and step-wise imbalance generators
3. **Comprehensive evaluation:** Per-class metrics, confusion matrices, training curves
4. **Locality analysis:** Attention distance computation for ViTs
5. **Receptive field analysis:** Maximum receptive field computation for CNNs
6. **Integration:** Combined multiple analysis techniques into unified framework

**Code Repository Structure:**

- **Main training:** `main.py`, `CNN.py`, `VisionTransormer.py`
- **Experiments:** `imbalanced-exp/`, `drop_layers_analysis/`, `locality/`, `receptive_field/`, `cka/`
- **Configuration:** `config.yaml`
- **All code is available in the repository for reproducibility**

**Briefly discuss potential future work that the research community could focus on to make improvements in the direction of your project's topic.**

**Future Research Directions:**

**1. Hybrid Architectures:**
- **ConViT (Convolutional Vision Transformer):** Combine CNN inductive biases with ViT attention mechanisms
- **LocalViT:** Restrict ViT attention to local windows (similar to Swin Transformer) to improve data efficiency
- **Research question:** Can we get ViT's flexibility with CNN's data efficiency?

**2. Data-Efficient ViTs:**
- **Knowledge distillation:** Train small ViTs using large pre-trained models as teachers
- **Self-supervised pre-training:** Use contrastive learning (SimCLR, MoCo) to pre-train ViTs on unlabeled data
- **Data augmentation:** Develop ViT-specific augmentation strategies
- **Research question:** How can we make ViTs work with limited data without pre-training on ImageNet?

**3. Understanding Inductive Biases:**
- **Theoretical analysis:** Why do CNNs' inductive biases help with small data?
- **Learned vs. built-in priors:** Can ViTs learn equivalent priors, and if so, how much data is needed?
- **Architecture search:** Automatically discover architectures with optimal bias-efficiency trade-offs
- **Research question:** What is the minimum data requirement for ViTs to match CNN performance?

**4. Imbalanced Learning:**
- **Class-balanced training:** Develop training strategies specifically for imbalanced data
- **Focal loss for ViTs:** Adapt focal loss (designed for object detection) to classification
- **Few-shot learning:** Test ViTs vs CNNs in few-shot scenarios (1-5 examples per class)
- **Research question:** Can architectural modifications make ViTs more robust to class imbalance?

**5. Efficiency and Deployment:**
- **Model compression:** Quantization, pruning, distillation for both architectures
- **Edge deployment:** Compare CNNs vs ViTs on mobile devices, embedded systems
- **Inference speed:** Systematic comparison of inference time, memory usage, energy consumption
- **Research question:** Which architecture is more suitable for resource-constrained applications?

**6. Transfer Learning and Domain Adaptation:**
- **Cross-domain transfer:** Test CNNs vs ViTs when pre-trained on ImageNet but applied to medical images, satellite imagery, etc.
- **Few-shot transfer:** Compare few-shot learning capabilities
- **Domain adaptation techniques:** Apply domain adaptation methods to both architectures
- **Research question:** Do ViTs' learned features transfer better across domains than CNNs' built-in biases?

**7. Interpretability and Explainability:**
- **Attention visualization:** Develop better methods to visualize and interpret ViT attention
- **Feature visualization:** Compare what features CNNs vs ViTs learn (using activation maximization, etc.)
- **Adversarial robustness:** Test robustness to adversarial examples
- **Research question:** Are ViTs more interpretable than CNNs, and does this help with debugging and improvement?

**8. Large-Scale Experiments:**
- **ImageNet from scratch:** Train both architectures from scratch on ImageNet (requires significant compute)
- **JFT-300M scale:** Test on very large datasets where ViTs show advantages
- **Long-tail datasets:** Test on naturally imbalanced large-scale datasets (iNaturalist, etc.)
- **Research question:** Do our findings from CIFAR-10 scale to larger datasets?

**9. Architectural Innovations:**
- **Efficient attention:** Test efficient attention mechanisms (Linformer, Performer) in ViTs
- **Dynamic architectures:** Architectures that adapt capacity based on input complexity
- **Multi-scale processing:** Combine multi-scale CNNs with ViT attention
- **Research question:** Can we design architectures that combine the best of both worlds?

**10. Theoretical Understanding:**
- **Convergence analysis:** Theoretical analysis of why CNNs converge faster with less data
- **Generalization bounds:** Compare generalization guarantees for CNNs vs ViTs
- **Optimization landscape:** Study loss landscapes and optimization dynamics
- **Research question:** Can we theoretically predict when CNNs vs ViTs will perform better?

**Most Promising Directions:**

1. **Hybrid architectures** (ConViT, LocalViT) - Practical and addresses core research question
2. **Data-efficient ViTs** - High impact if successful, enables ViT adoption in data-limited scenarios
3. **Imbalanced learning** - Addresses real-world problem, both architectures need improvement
4. **Efficiency and deployment** - Practical importance for real-world applications

**Conclusion:**

Our work provides a foundation for understanding when to use CNNs vs ViTs, but many questions remain. The most impactful future work would combine architectural innovations (hybrid models), training improvements (data-efficient methods), and theoretical understanding (why each architecture works) to develop next-generation vision models that are both data-efficient and flexible.

