# Pathloss Prediction With Satellite Image




## Background

Radio wave propagation in dense urban areas is extremely susceptible to environmental influences, and path loss is easily affected by the surrounding ground environment and fluctuates greatly. An enhanced pathloss prediction method is explored using digital and satellite maps.
<div align="center">
<img src="https://github.com/zhaozhang101/Pathloss-Prediction-with-Satellite-Image/assets/71812547/1b7f0067-c4e7-438c-bdaa-40edfc047988" width="600px"></div>
<div align="center">
<img src="https://github.com/zhaozhang101/Pathloss-Prediction-with-Satellite-Image/assets/71812547/de32e6ac-95b2-4628-8281-806dd92d6ad0" width="1000px"></div>

The above two pictures are digital maps and satellite maps of the relevant area. Based on the road test data of Hangzhou, Ningbo, Wenzhou (urban areas) and the suburbs of Malaysian cities and towns, we explore the gain of satellite images in pathloss prediction.
<div align="center">
<img src="https://github.com/zhaozhang101/Pathloss-Prediction-with-Satellite-Image/assets/71812547/337e1f35-d31c-4e34-9175-95764a463540" width="600px"></div>

The fitting results by 'log' of the four cities are shown in the figure above. The measured data in suburban areas are more consistent with the traditional propagation model.

<div align="center">
<img src="https://github.com/zhaozhang101/Pathloss-Prediction-with-Satellite-Image/assets/71812547/baa4d1ce-42e7-44f2-9f3d-b4d0cdf94203" width="600px"></div>
The performance of fitting/predicting using different methods is shown in the figure above. For the ANN method, input information includes the cell index, the azimuth angle of the receiver relative to the base station, and the distance between the receiver and the base station. Note that adding satellite images has limited improvement in the results. This may be due to the following reasons: 

• The resolution of satellite images is too low to give detailed information. 

• The method of encoding satellite images needs to be improved, especially for longer Tx-Rx link distances, crucial information could be lost using the 'image-resize' approach.

• The original measurement data contains noise due to general small-scale fading which means low data quality.

## Install
The necessary packages are given in requirement.txt.

## Usage
Download the projects to check in detail.

## Dataset
Due to confidentiality reasons, the data will not be publicly downloaded. Please contact us by email (`zhaozhang@bjtu.edu.cn`) if interested.

## Contributors
This project exists thanks to all the people who contribute.

