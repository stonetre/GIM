SWaT.A1 & A2_Dec 2015 dataset contains time series of normal operation and fault operation. 
Delete the device state variables and concatenate the csv file of the fault dataset to the end of the csv dataset of the normal operation data.
Add a Label column to the last column, with 1 for fault and 0 for normal. 
Rename the sorted dataset to node_properties.csv and put it in the directory.
Adjust the parameter settings in the code:
<pre><code>
args.exp_name = 'SWAT2015_V15'
args.datadir = 'data/SWAT2015_V15'
args.datasets = ['SWAT2015_V15']
args.train_percent = 0.42
args.valid_percent = 0.11
args.test_percent = 0.47
args.timeseries_downsample = 10
args.batch_size =200
args.sib_k = 1
args.max_node_num = 15
args.gsl_module_type = 'gim'
args.gsl_use_prior_learner = True
args.gsl_use_description_learner = False
args.gsl_feature_len = 60  
args.gsl_feature_dim = 1
args.gsl_interact_type = 'mamba'
args.gsl_feature_hidden_dim = 16
args.encoder_feature_len = 60
args.encoder_layer_type = 'mamba'
args.encoder_feature_layer = 1
args.encoder_hidden_dim = 64
args.encoder_out_len = 60
args.encoder_out_dim = 64
args.pred_decoder_type = 'gim'
args.pred_decoder_interact_type = 'scm'                    
args.pred_decoder_feature_layers = 1
args.pred_decoder_feature_hidden_dim = 64 
args.pred_decoder_predict_len = 20
args.pred_decoder_predict_dim = 1
</code></pre>
