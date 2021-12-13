#!/bin/bash

export CUDA_VISIBLE_DEVICES=

#tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model --control_flow_v2=True ./saved_models/MLP.h5 ./web-client/saved_models/graphs/MLP
#tensorflowjs_converter --input_format keras --output_format=tfjs_graph_model --control_flow_v2=True ./saved_models/CNN.h5 ./web-client/saved_models/graphs/CNN

#tensorflowjs_converter --input_format keras ./saved_models/MLP.h5 ./web-client/saved_models/layers/MLP
#tensorflowjs_converter --input_format keras ./saved_models/CNN.h5 ./web-client/saved_models/layers/CNN


tensorflowjs_converter --output_format=tfjs_graph_model --control_flow_v2=True ./saved_models/MLP ./web-client/saved_models/graphs/MLP
tensorflowjs_converter --output_format=tfjs_graph_model --control_flow_v2=True ./saved_models/CNN ./web-client/saved_models/graphs/CNN

#tensorflowjs_converter ./saved_models/MLP ./web-client/saved_models/layers/MLP
#tensorflowjs_converter ./saved_models/CNN ./web-client/saved_models/layers/CNN
