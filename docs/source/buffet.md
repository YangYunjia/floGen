[TOC]

# Buffet onset estimation

Transonic buffet is an dangerous phenomenon happens on the upper surface of the supercritical airfoils, so it is very important to predict the buffet onset (the angle of attack that the buffet happens) for the airfoils. Although transonic buffet is an unsteady phenonmenon, it can be predicted with engineering method based on airfoil's aerodynamic curves. The curves including the lift curve (the curve of the lift coefficients v.s. the angles of attack) and the pitching moment curve.

Here, we use the FloGen model as a off-design flowfield generator to predict the aerodynamic curves from the cruise flowfield. The FloGen provide several new Classes to master that job, they are inside `flowvae.app.buffet`, and we also have some examples avilable in `examples/buffet` for establishing dataset, training model, and using the model to predict buffet onset. In the following, we will introduce the Classes in `buffet.py`

## Collecting the series of aerodynamic coefficients

The support class to store a series of aerodynamic coefficients for the prediction of the buffet onset.

For each airfoil, we can construct a `Series` class:

```python
seri_r = Series(['AoA', 'Cl', 'Cd'])
```

This means the series as three aerodynamic variables: the angle of attacks (`AoA`), the lift coefficient (`Cl`) and the drag coefficient (`Cd`).

> **Remark**: In `Series`, the key `AoA` is seen as the main key, and must be assigned when initialization.

Then, after we predicting the aerodynamic coefficients of the airfoil under an angle of attack, we can use the following code to add it to the series:

```python
seri_r.add(x={'AoA': aoa_r, 'Cl': cl_r, 'Cd': cd_r})
```

The `add` function of `Series` will **automatically** sort the input values with the key `AoA`.

If we already have a array of the coefficients, we can directly assign it to the series with:

```python
seri_r = Series(['AoA', 'Cl', 'Cd'], datas={'AoA': aoas, 'Cl': clss, 'Cd', cdss})
```

where `aoas`, `clss`, and `cdss` should be  `np.array`. The sorting will not automatically called in this case, if needed, use

```python
seri_r.sort()
```

## Estimating the buffet onset

With the series of aerodynamic coefficients, we can predict the buffet onset with a buffet-onset-estimitor defined as class `Buffet`. It should be initialized with

```python
buffet_cri = Buffet(method='lift_curve_break', lslmtd=2, lsumtd='error', lsuth2=0.01, intp='1d')
```

The argument `method` defines how to estimate the buffet onset, and the rest coefficients defines the parameters of this method.

Then, we can predict the onset with:

```python
buf_r = buffet_cri.buffet_onset(seri_r, cl_c=cl_cruise)
```

The first argument is the series data of the airfoil, and there may also be some parameters that is related to the airfoil itself. They should be input to the function as well.

The avilable buffet onset estimation method include:

|name|description|
|-|-|
|`lift_curve_break`|lift curve break method, estimate the buffet onset with the intersection between the lift curve and the shifted linear section of the lift curve
|`adaptive_lift_curve_break`|adaptive method to decide the angle of attack to simulate and give the buffet onset with the lift curve break method|
|`curve_slope_diff`| estimate the buffet onset when the slope of the given curve (i.e., the lift curve or pitching moment curve) changes a given value according to the linear section|
|`cruve_critical_value`| estimate the buffet onset at the maximum (or the minimum) of a curve  (i.e., the pitching moment curve or the curvature curve of the pitching moment)
|`shock_to_maxcuv`| estimate the buffet onset when the shock wave on upper surface reach the maximum curvature point of the airfoil geometry|