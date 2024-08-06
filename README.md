# GIM
This project is mainly used for graph structure learning of real industrial processes. 
## Multiple dataset support
The code supports multi-dataset training, which is particularly useful when real industrial engineering data is scarce. Researchers can collect datasets that are similar to industrial processes, construct standardized datasets, and train models directly using the provided code. The code utilizes a node mask matrix to manage the varying numbers of sensors across different datasets. However, a hyperparameter specifying the maximum number of nodes needs to be determined for all datasets to ensure consistency across the training process.

## Multimodal data support
The code also provides support for incorporating node descriptions. If your data is collected from a Distributed Control System (DCS), it is likely that you will obtain data that includes descriptions of the nodes' functions, expressed in natural language. Our code utilizes SentenceTransformers as word2vec encoders to process these descriptions. During preprocessing, this encoded data is saved directly into the dataset. Although this paper does not leverage this information, researchers have the option to use it when building their own models.

## Dataset construction
Pre-provided data inputs include: node_properties.csv, node_description.csv, graph.csv
node_properties.csv is the time series data of the node. The first column of the data is the sampling timestamp. The first line is the sensor name.
node_description.csv is used to describe the text information of each sensor. The sensor name in the first column must correspond to the name in node_properties.csv. It must contain the sensor type column, which is the necessary description information. Other additional information will be spliced ​​during processing. 
graph.csv contains the edge indices of the graph. The indices point from the sensors in the first column to the sensors in the second column. Note that the order of the node indices needs to be consistent with the order in node_properties.csv. 

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
