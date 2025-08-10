# GIM
This project is mainly used for graph structure learning of real industrial processes. Paper: Spatial information bottleneck graph structure learning based multivariate time series prediction for industrial processes
![Model framework](https://github.com/stonetre/GIM/blob/main/pic/framwork.jpg?raw=true)

## Dataset construction
Pre-provided data inputs include: node_properties.csv, node_description.csv, graph.csv

node_properties.csv is the time series data of the node. The first column of the data is the sampling timestamp. The first line is the sensor name.

node_description.csv is used to describe the text information of each sensor. The sensor name in the first column must correspond to the name in node_properties.csv. It must contain the sensor type column, which is the necessary description information. Other additional information will be spliced ​​during processing. 

graph.csv contains the edge indices of the graph. The indices point from the sensors in the first column to the sensors in the second column. Note that the order of the node indices needs to be consistent with the order in node_properties.csv. 

