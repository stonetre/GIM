# GIM
Industrial graph structure learning
## Multiple dataset support
The code supports multi-dataset training, which is particularly useful when real industrial engineering data is scarce. Researchers can collect datasets that are similar to industrial processes, construct standardized datasets, and train models directly using the provided code. The code utilizes a node mask matrix to manage the varying numbers of sensors across different datasets. However, a hyperparameter specifying the maximum number of nodes needs to be determined for all datasets to ensure consistency across the training process.

## Multimodal data support
The code also provides support for incorporating node descriptions. If your data is collected from a Distributed Control System (DCS), it is likely that you will obtain data that includes descriptions of the nodes' functions, expressed in natural language. Our code utilizes SentenceTransformers as word2vec encoders to process these descriptions. During preprocessing, this encoded data is saved directly into the dataset. Although this paper does not leverage this information, researchers have the option to use it when building their own models.

## Test dataset results
The dynamic process inferred for time steps 4000-5000 on the training set is illustrated in the following animation. For a higher resolution video, click the link below.

<div style="text-align: center;">
    <img src="gifs/combined_best_pred_1K.gif" alt="dynamic graph" title="dynamic graph" />
</div>

Video of all test dataset:
<div style="text-align: center;">
    <a href="https://www.youtube.com/watch?v=WCwuvTBgeHk">
        <img src="https://img.youtube.com/vi/WCwuvTBgeHk/0.jpg" alt="Video Title" />
    </a>
</div>
