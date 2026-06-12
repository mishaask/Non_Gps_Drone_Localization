Literature Review: Visual Navigation for Drones in GNSS-Denied Conditions

**Assignment Context and Problem Definition**
  
    This project focuses on GNSS-denied visual navigation for drones. The assignment defines the main problem as estimating the drone’s position without using GNSS during the new/test flight.
    More specifically, the required task is to preprocess a reference drone flight that contains video and telemetry, including GNSS position,
    barometric height, and camera angle, and then use this preprocessed data to estimate, in real time, the geographic coordinate of the center point of a new drone video stream without GNSS.
    
    This requirement places the project between several related computer-vision fields: visual place recognition, visual geo-localization, local feature matching, geometric verification, visual SLAM, and aerial photogrammetry. 
    Pure SLAM can estimate relative motion, but it usually does not directly give global latitude/longitude unless aligned to a geo-referenced map, and didnt return any good results while testing the idea. 
    In contrast, visual geo-localization and visual place recognition are closer to the assignment, because they match a query drone frame to a database of geo-tagged reference images or map tiles (kinda how humans work). 
    The most suitable approach for this project is therefore a hybrid pipeline: build a geo-tagged reference database during preprocessing, 
    retrieve candidate reference frames for each GNSS-denied query frame, verify the match geometrically, and then project the video center point into geographic coordinates.

**Visual Place Recognition and Image Retrieval**
    
    Visual Place Recognition (VPR) is the task of identifying whether a query image shows a previously observed place. 
    This is highly relevant to GNSS-denied drone navigation because each query frame from the drone can be matched against a database of geo-tagged reference frames. 
    Classical VPR methods often used handcrafted local features, but recent work has moved toward deep global descriptors that are more robust to changes in viewpoint, lighting, scale, and scene appearance.

    A foundational deep VPR method is NetVLAD, which introduced a CNN architecture with a trainable VLAD pooling layer for weakly supervised place recognition. 
    NetVLAD is older than the newest papers, but it remains important because many later retrieval systems build on the idea of aggregating local visual features into a compact global image descriptor. 
    The original NetVLAD work targets large-scale place recognition, where a query photograph is compared against a database to identify its location. 
    Source: [NetVLAD paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html)
