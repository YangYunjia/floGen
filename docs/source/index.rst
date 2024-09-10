.. flowGen documentation master file, created by
   sphinx-quickstart on Mon Jul  1 23:56:46 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to flowGen's documentation!
===================================

FlowGen is a universal flowfield generator based on Auto-Encoder (AE) & Variational Auto-Encoder (VAE) and implentmented in Pytorch. 
It differs from other flowfield generators that it introduce **prior learning strategy** to enhance the predicting accuracy and transfer 
learning capability. The FlowGen has been applied to the prediction and optimization tasks such as the transonic buffet and the fluidic 
injection into the single expansion ramp nozzle (SERN). We also developed a web-based interactive optimization app for transonic wings
(Webwing).

**Author:** 

- Yunjia Yang, Tsinghua University, yyj980401@126.com (Main)
- Yuqi Cheng, yc4330@columbia.edu (UI for Webwing)

**Contributor (former user):** 

- Runze Li, lirunze16@tsinghua.org.cn
- Zuwei Tan (Supersonic 2D inlet)
- Gongyan Liu (Temperature field of a data center)
- Jiazhe Li (Supersonic 2D single expansion ramp nozzle)


Contents
==================

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   airfoildataset
   wingdataset

This section presents several flowfield datasets. Most of them are of airfoils and wings. 
They are available under reasonable requests. Please contact Yunjia Yang (yyj980401@126.com) 
for the datasets.

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   simpleguide
   models

This section describe the model.

.. toctree::
   :maxdepth: 2
   :caption: Applications

   buffet
   wing

This section provides some applications of the FloGen. You can find the corresponding `.py` 
files in `examples`, and the data files can be obtained by communicating with the author.