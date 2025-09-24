
# FloGen

FlowGen is a universal flowfield generator based on machine learning and implentmented in Pytorch. 
- It differs from other data-driven flowfield generators that it introduce **physics knowledge** to enhance the predicting accuracy and generaliztion capability. 
- It has been applied to the prediction of airfoil, [wing](#web-wing---interactive-transonic-wing-design-app) and single expansion ramp nozzle (SERN).
- It is successfull applied to gradient-free optimization and gradient-based optimization (with utilizing back-propagation) tasks such as optimizing transonic buffet onset and multi-design-point fluidic injection parameters.

You can find the documentation of flowGen [here](https://flogen.readthedocs.io/en/latest/).

**Authors:** 

- Yunjia Yang, Tsinghua University, yyj980401@126.com
- Yuqi Cheng, yc4330@columbia.edu (UI for Webwing)

**Thanks to:** 

- Runze Li, lirunze16@tsinghua.org.cn
- Mengxin Liu, for wing dataset estabilishment

**Citation:**

- multipoint airfoil flow field prediction
    ```
    Yang, Yunjia, Runze Li, Yufei Zhang, and Haixin Chen*. 2022. “Flowfield Prediction of Airfoil Off-Design Conditions Based on a Modified Variational Autoencoder.” AIAA Journal 60 (10): 5805–20. https://doi.org/10.2514/1.J061972.
    ```

- wing flow field prediction
    ```
    Yang, Yunjia, Runze Li, Yufei Zhang, Lu Lu, and Haixin Chen*. 2024. “Transferable Machine Learning Model for the Aerodynamic Prediction of Swept Wings.” Physics of Fluids 36 (7): 076105. https://doi.org/10.1063/5.0213830.
    ```

## Datasets

The flowfield datasets to train all of our models are available upon request. Please contact Yunjia Yang (yyj980401@126.com) for the datasets. Details on the datasets can be found [here](https://flogen.readthedocs.io/en/latest/).

## Applications

### Web-wing - Interactive Transonic Wing Design App

The physics-embedded transfer learning for transonic wing is demonstrated with a simple interactive app at [TUM website](https://webwing.pbs.cit.tum.de/). You can modify the airfoil geometry, wing planform geometry, and wing operating conditions to see what will happen on the wing surface flow field. Feel free to play with it, and your knowledge on wing aerodynamics will grow.

You can also locally deploy it, and you can find how to do it [here](https://github.com/YangYunjia/webwing).

The next step of the app is a gradient optimization tool for wing performance, which will come soon.

![](docs/source/_static/images/webwing/webwing.gif)