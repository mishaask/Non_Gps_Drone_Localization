***Literature Review: Visual Navigation for Drones in GNSS-Denied Conditions***

**Assignment Context and Problem Definition**
  
  This project focuses on GNSS-denied visual navigation for drones. The assignment defines the main problem as estimating the drone’s position without using GNSS during the new/test flight.
  More specifically, the required task is to preprocess a reference drone flight that contains video and telemetry, including GNSS position,
  barometric height, and camera angle, and then use this preprocessed data to estimate, in real time, the geographic coordinate of the center point of a new drone video stream without GNSS.
    
  This requirement places the project between several related computer-vision fields: visual place recognition, visual geo-localization, local feature matching, geometric verification, visual SLAM, and aerial photogrammetry. 
  Pure SLAM can estimate relative motion, but it usually does not directly give global latitude/longitude unless aligned to a geo-referenced map, and didnt return any good results while testing the idea. 
  In contrast, visual geo-localization and visual place recognition are closer to the assignment, because they match a query drone frame to a database of geo-tagged reference images or map tiles (kinda how humans work). 
  The most suitable approach for this project is therefore a hybrid pipeline: build a geo-tagged reference database during preprocessing, 
  retrieve candidate reference frames for each GNSS-denied query frame, verify the match geometrically, and then project the video center point into geographic coordinates.

  Our implemented prototype follows a hierarchical visual localization pipeline. 
  ->First, reference drone videos with SRT telemetry are preprocessed into geo-tagged reference frames. 
  ->Then, for each query frame, the system performs global visual retrieval using an AnyLoc/DINOv2-style descriptor backend. 
  ->The top candidate reference frames are then verified with LightGlue local feature matching and geometric filtering using homography and inlier checks. 
  ->Accepted matches are converted into predicted geographic coordinates and exported as CSV/KML paths.

  This literature review therefore separates the reviewed methods into three categories: methods directly integrated into the current pipeline, methods used as conceptual support, 
  and methods reviewed but not selected because they do not fit the assignment requirements as directly.

**Visual Place Recognition and Image Retrieval**
    
  Visual Place Recognition (VPR) is the task of identifying whether a query image shows a place that was previously observed. 
  This is central to our project because each GNSS-denied query frame must be compared against a database of GNSS-tagged reference frames. 
  Instead of estimating only relative motion, VPR allows the system to recover an absolute position by finding the most visually similar known location.

  Older VPR methods such as NetVLAD are important background. NetVLAD introduced a trainable VLAD pooling layer inside a CNN architecture for weakly supervised place recognition. 
  It showed that local visual information could be aggregated into a compact global image descriptor for large-scale place recognition.
  NetVLAD is therefore relevant as a foundational baseline, but it is not used in our current pipeline because it is older and usually depends more on task-specific training or datasets. 
  Our project instead needed a practical descriptor that could work on our drone footage without training a new model.
  Source: [NetVLAD paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html)

  CosPlace is another important visual geo-localization approach. It reformulates visual geo-localization as a classification problem, making training scalable for large-scale map-level localization. 
  It is relevant because our system also needs to retrieve the correct visual location from a reference database. 
  However, CosPlace is not directly used in our implementation because the current assignment prototype is built around pretrained general-purpose features and reference-frame indexing rather than training a new large-scale geo-localization classifier.
  This makes AnyLoc/DINOv2 more suitable for our available time, data, and project constraints.
  Source: [CosPlace Github](https://github.com/gmberton/cosplace)

 *The most relevant recent VPR direction for this project is AnyLoc.* 
  AnyLoc proposes a universal visual place-recognition approach using general-purpose self-supervised visual features, especially DINOv2, combined with unsupervised aggregation. 
  The authors specifically target broad deployment across many different environments, including aerial environments, without retraining or fine-tuning. 
  This is directly aligned with our drone videos, where reference and query frames can differ in altitude, camera angle, scale, lighting, and viewpoint.
  Source: [Anylock Paper](https://arxiv.org/abs/2308.00688)

  For this reason, our current pipeline uses an AnyLoc-style retrieval stage, implemented through the anyloc-gem descriptor backend. 
  This is the first major decision in the pipeline: before doing expensive local matching, the system quickly retrieves the most likely reference frames. 
  This makes the pipeline more efficient because LightGlue does not need to compare every query frame against every reference frame.

**DINOv2 as the Visual Feature Backbone**
  A key reason AnyLoc is suitable for this project is its use of strong general-purpose visual features. DINOv2 is a self-supervised visual foundation model designed to produce robust visual features without manual labels.
  The DINOv2 paper argues that pretrained self-supervised models can produce all-purpose features that transfer across image distributions and tasks, especially when trained on large curated datasets. 
  Source: [DINOv2 Paper](https://arxiv.org/abs/2304.07193)

  DINOv2 is useful in our project because drone localization contains several difficult visual changes. The same location may appear from a different height, from a slightly different heading, at a different scale, or under different lighting. 
  Traditional handcrafted features such as ORB can work well when images have strong repeatable corners and similar viewpoints, but they are less reliable when the viewpoint or scale changes significantly. 
  DINOv2-based descriptors encode more semantic and structural information, making them stronger for the global retrieval stage.

  In our pipeline, DINOv2 is not used as a standalone localization system. It is used as the feature backbone inside the AnyLoc-style descriptor stage. 
  This means DINOv2 helps answer the question: “Which reference frames are visually most similar to this query frame?” After that, the candidate matches still need to pass local feature matching and geometric verification. 
  This division is important because DINOv2/AnyLoc gives strong candidate retrieval, but it does not by itself prove that the query and reference frame are geometrically consistent.

**Multi-Scale Retrieval for Drone Height and Viewpoint Differences (This is my idea)**

  One practical challenge in our project is that the query videos are not always recorded from exactly the same height as the reference videos. 
  Our earlier experiments used scale values such as 0.2 and 0.3, and later considered larger multi-scale sets such as 0.15, 0.2, 0.3, 0.4. 
  The current project draft notes that the implementation uses multi-scale retrieval together with AnyLoc/DINOv2 and LightGlue verification.

  This multi-scale step is important because low-flying drone imagery is strongly affected by altitude. A road, building, field, or intersection can occupy a very different number of pixels depending on the drone’s height. 
  If the query frame is higher than the reference frame, direct matching at one fixed scale may fail. Multi-scale retrieval partially solves this by testing resized versions of the query frame, 
  increasing the chance that the visual content appears at a comparable scale to the reference database.

**Local Feature Matching and Geometric Verification**

  Global retrieval is necessary but not sufficient. A descriptor such as AnyLoc/DINOv2 can retrieve visually similar places, but drone footage often contains repeated patterns: roads, fields, rooftops, parking areas, and similar intersections. 
  Therefore, the retrieved candidate may look similar but still be geographically wrong. To reduce false positives, our system applies local feature matching and geometric verification after retrieval.

  The main local matcher used in our current implementation is LightGlue. LightGlue is a learned local feature matcher designed to match sparse features efficiently. 
  It improves on the SuperGlue-style matching family by being faster and more adaptive: easier image pairs require less computation, while harder pairs receive more processing.
  This is useful for our project because a real-time or near-real-time drone pipeline must avoid spending too much time on every candidate pair.
  Source: [LightGlue paper](https://arxiv.org/abs/2306.13643)

  In our pipeline, LightGlue is used after AnyLoc/DINOv2 retrieval. The role of LightGlue is not to search the whole database. 
  Instead, it verifies whether a small number of retrieved candidate frames are actually consistent with the query frame. This is a key design choice: AnyLoc/DINOv2 gives recall, while LightGlue gives precision.

  After LightGlue produces matches, the system applies geometric filtering using homography and inlier-based checks. This stage rejects candidates when there are not enough good matches, not enough inliers, a weak inlier ratio, or no valid homography. 
  The uploaded project draft specifically mentions rejection cases such as not_enough_good_matches, not_enough_inliers, homography_not_found, and homography/inlier-based filtering.

  This geometric step is essential because it connects the literature to the final coordinate output. Once the query frame and reference frame are geometrically aligned, the query image center can be projected into the matched reference frame.
  Since the reference frame is associated with telemetry, this projected point can then be converted into a predicted geographic coordinate.

**LoFTR as an Alternative Matcher (Reviewed but not integrated)**

  LoFTR is another important learned matching method. Unlike detector-based pipelines, LoFTR is detector-free: it establishes coarse dense correspondences first and then refines them. 
  This can help in low-texture areas where traditional keypoint detectors struggle. Drone imagery often contains fields, roads, roofs, and other surfaces with weak or repetitive local texture, so LoFTR is relevant to this project.

  However, LoFTR is not the main matcher in our current pipeline. We selected LightGlue because the current implementation already works around sparse local verification after retrieval, and LightGlue is designed for efficient matching.
  LoFTR remains a strong future improvement or fallback option for cases where LightGlue rejects too many frames due to insufficient keypoint matches, especially over low-texture terrain.

  Source: [LoFTR Paper](https://arxiv.org/abs/2104.00680)


**UAV-Specific Visual Localization Research**

  Several recent UAV-specific works directly support our project direction. WildNav proposes vision-based GNSS-free localization for UAVs in outdoor environments.
  It matches RGB images from a drone against a pre-built map made from geo-referenced satellite images and computes the UAV’s geographic coordinates using deep features. 
  This is highly relevant because it frames UAV navigation as a visual localization problem under GNSS-free conditions.

  WildNav is not directly used in our implementation because its reference source is different. 
  WildNav localizes drone imagery against satellite/open-source maps, while our current system localizes a query drone video against reference drone video frames with SRT telemetry.
  Nevertheless, WildNav strongly supports our choice to use visual matching for GNSS-denied UAV localization and might be a strong toolto use when we handle our final project(google assisted no GNSS localization).

  Source: [Vision-based GNSS-Free Localization for UAVs in the Wild](https://arxiv.org/abs/2210.09727)

  UAV-VisLoc is another relevant contribution. It defines UAV visual localization as determining a UAV’s latitude and longitude by matching the UAV’s ground-looking image to satellite maps. 
  The dataset includes drone images, satellite maps, and metadata such as latitude, longitude, altitude, and capture date. This supports the idea that UAV localization can be treated as matching query drone imagery to geo-referenced visual data.

  UAV-VisLoc is not used as our dataset because the assignment requires working with the provided DJI/Autel videos and their telemetry.
  However, it is valuable related work because it shows that the research community evaluates UAV localization using image-to-map matching and geographic ground truth.

  Source: [UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization](https://arxiv.org/abs/2405.11936)

  The UAV-AVL / AnyVisLoc benchmark is especially relevant because it focuses on low-altitude multi-view UAV absolute visual localization. 
  The benchmark defines UAV absolute visual localization as determining the UAV position in GNSS-denied conditions by establishing geometric relationships between UAV images and geo-tagged reference maps.
  It also emphasizes that low-altitude multi-view conditions are difficult because of extreme viewpoint changes. This directly matches the challenges in our project, where query and reference drone frames may differ in height, angle, and viewpoint.

  UAV-AVL is not directly integrated because it is a benchmark and dataset framework, while our task is to implement a solution for the assignment videos. 
  Still, it justifies our core design: retrieval alone is not enough; the system needs retrieval, matching, and geometric verification.

  Source: [Exploring the best way for UAV visual localization under Low-altitude Multi-view Observation Condition: a Benchmark](https://arxiv.org/abs/2503.10692)

**Visual SLAM and Visual-Inertial SLAM**

  Visual SLAM methods estimate camera motion and reconstruct a local map from video.
  ORB-SLAM3 is a major open-source SLAM system that supports visual, visual-inertial, and multi-map SLAM with monocular, stereo, and RGB-D cameras. It is highly relevant to robotics and drone navigation because it can estimate camera trajectory in real time.

  However, SLAM is not the main method in our pipeline. The reason is that SLAM mostly estimates a relative trajectory.
  It can tell how the camera moved relative to its starting point, but it does not automatically provide latitude and longitude unless the SLAM map is aligned to a geo-referenced coordinate system. 
  The assignment requires the geographic coordinate of the center point of the video frame, so pure SLAM does not directly solve the required output.

  Therefore, ORB-SLAM3 is reviewed as important related work but not selected as the core implementation. 
  They may be useful in future work for temporal smoothing, motion consistency, or stabilizing the predicted path, but the main implemented solution remains reference-based visual geo-localization.
  Source: ORB-SLAM3: [An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM](https://arxiv.org/abs/2007.11898)

**Photogrammetry and 3D Reconstruction Tools to try and build a "refrence map"**

  For preprocessing, tools such as COLMAP and OpenDroneMap were also considered. COLMAP is a general-purpose Structure-from-Motion and Multi-View Stereo pipeline that can reconstruct camera poses and 3D scene structure from overlapping images. 
  Structure-from-Motion is generally used to estimate 3D structure from 2D image sequences, and COLMAP is one of the common open-source tools in this area.

  OpenDroneMap is specifically designed for drone imagery. It can process aerial imagery into maps and 3D models, including orthomosaics, point clouds, and georeferenced outputs. 
  This is relevant because a stronger future version of the project could create an orthomosaic or 3D reference map from the GNSS-tagged reference flight.

  However, neither COLMAP nor OpenDroneMap is used in the current pipeline. The reason is practical: they add a heavier offline reconstruction stage.
  The assignment can be solved more directly by extracting reference frames, parsing SRT telemetry, building a descriptor database, retrieving candidates, verifying them, and exporting predicted coordinates.
  This simpler approach is easier to document, easier to run, and closer to the current project implementation.

  Source: [Structure from motion](https://en.wikipedia.org/wiki/Structure_from_motion) and [OpenDroneMap](https://en.wikipedia.org/wiki/OpenDroneMap)

**Object Detection and Segmentation Methods**

Object detection and segmentation methods such as YOLO, SAM, FastSAM, SAM2, SegFormer, Mask2Former, and DeepLab-style models were considered as possible supporting tools, but they are not part of the current localization pipeline.
Source: my Final project revolves around yolo and sam

**Conclusion**
  The literature shows that GNSS-denied drone navigation should be treated primarily as an absolute visual localization problem. Pure SLAM systems such as ORB-SLAM3 and DROID-SLAM are powerful for relative motion estimation,
  but they do not directly provide global latitude/longitude without geo-referenced alignment. Object detection and segmentation models can provide useful semantic information, but they also do not directly solve geographic localization.

  The most appropriate direction for this project is therefore a hierarchical visual localization pipeline. Recent VPR methods such as AnyLoc and foundation features such as DINOv2 support robust candidate retrieval under viewpoint, altitude,
  and appearance changes. LightGlue provides efficient local verification of retrieved candidates, while homography and inlier filtering reject false matches and allow the query frame center to be projected into the reference frame.
  UAV-specific works such as WildNav, UAV-VisLoc, and UAV-AVL support the overall idea that GNSS-denied UAV localization can be solved by matching drone imagery against geo-referenced visual data,
  although their datasets and reference sources differ from our assignment videos.

***Selected Direction and Integration in Our Pipeline***

    GNSS-tagged reference drone videos + SRT telemetry
            ↓
    Extract reference frames and parse metadata
            ↓
    Build visual descriptor database using AnyLoc-GEM / DINOv2
            ↓
    Take query frames from the test drone video
            ↓
    Treat query video GNSS as unavailable during prediction
            ↓
    Run multi-scale retrieval to find candidate reference frames
            ↓
    Verify candidates using LightGlue
            ↓
    Apply homography and inlier filtering
            ↓
    Use the accepted matched reference frame to estimate the query center location
            ↓
    Convert accepted matches into latitude/longitude predictions
            ↓
    Export CSV and KML predicted path
            ↓
    Compare predicted path against the query video SRT/GNSS ground truth for evaluation
