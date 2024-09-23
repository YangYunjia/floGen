
# FloGen

FlowGen is a universal flowfield generator based on Auto-Encoder (AE) & Variational Auto-Encoder (VAE) and implentmented in Pytorch. It differs from other flowfield generators that it introduce **prior learning strategy** to enhance the predicting accuracy and transfer learning capability. The FlowGen has been applied to the prediction and optimization tasks such as the transonic buffet and the fluidic injection into the single expansion ramp nozzle (SERN).

You can find the documentation of flowGen [here](https://flogen.readthedocs.io/en/latest/).

**Author:** 

Yunjia Yang, Tsinghua University, yyj980401@126.com
Yuqi Cheng, yc4330@columbia.edu, UI for Webwing

**Contributor (former user):** 

- Runze Li, lirunze16@tsinghua.org.cn
- Zuwei Tan (Supersonic 2D inlet)
- Gongyan Liu (Temperature field of a data center)
- Jiazhe Li (Supersonic 2D single expansion ramp nozzle)

**Citation:**

- multipoint airfoil flow field prediction
    ```
    Yunjia Yang, Runze Li, Yufei Zhang, and Haixin Chen*. 2022. “Flowfield Prediction of Airfoil Off-Design Conditions Based on a Modified Variational Autoencoder.” AIAA Journal 60 (10): 5805–20. https://doi.org/10.2514/1.J061972.
    ```

- wing flow field prediction
    ```
    Yang, Yunjia, Runze Li, Yufei Zhang, Lu Lu, and Haixin Chen*. 2024. “Transferable Machine Learning Model for the Aerodynamic Prediction of Swept Wings.” Physics of Fluids 36 (7): 076105. https://doi.org/10.1063/5.0213830.
    ```

## Motivations

During aerodynamic shape optimization (ASO), it is usually not enough to only optimize the **design point performance** since it may cause undesired performance losses at **off-design points**. Therefore, an important task of ASO is to fast and accurately predict a series of flowfields and their aerodynamic performances. 

The compuatational fluid dynamic (CFD) method is usually used to obtain such a series of flowfield, yet it leads to large compuation cost and may not be a wise choise during ASO which needs each sample to be evaluated fast. Thus, we can leverage the capability of the deep nerual network (DNN) and construct a DNN-based surrogate model to calculate the massive off-design flowfields.

In order to improve the prediction accuracy, the FlowGen is designed to predict a series of off-design flowfields with a **prior flowfield** as reference. The advantages are two-folded.

![](docs/source/_static/images/flowgen1.png "Predicting a series of off-design flowfield with the prior flowfield")

### Introducing the *Prior*

Normally, the design point and off-design points flowfields have strong relationships. It is because that the variation of the flowfield with the design parameters (i.e., the freestream Mach number) is continuous and the flow structures maintain the same. 

Therefore, **introducing the design flowfield as the input** (in the statistic perspective, as the *prior*) **when generating the off-design flowfields should improve model's perfromance**.

### Residual learning

Another motivation to introducing design flowfield when predicing off-design flowfield is from the common understand of the deep neural network. Since the ResNet, people has know that DNN is easier to learn a 0 - 0 mapping than a non-zero identical learning. In another word, the model's performance can be improved by subtracing the same part. 

Therefore, there's an option in FlowGen to make the model predicting the **difference** (or the residual) **between the target flowfield** (in most time, the off-design flowfield) **and the prior flowfield** (the design flowfield). 

## Datasets

The flowfield datasets to train all of our models are available upon request. Please contact Yunjia Yang (yyj980401@126.com) for the datasets. Details on the datasets can be found [here](https://flogen.readthedocs.io/en/latest/).

## Applications

### Web-wing - Interactive Transonic Wing Design App

The physics-embedded transfer learning for transonic wing is demonstrated with a simple interactive app. You can modify the airfoil geometry, wing planform geometry, and wing operating conditions to see what will happen on the wing surface flow field. Feel free to play with it, and your knowledge on wing aerodynamics will grow.

Unfortunately, the author can not afford a server now, so the only way access to web-wing is running locally. You can find how to do it [here](https://flogen.readthedocs.io/en/latest/wingapp.html).

The next step of the app is a gradient optimization tool for wing performance, which will come soon.

![](docs/source/_static/images/webwing/webwing.gif)