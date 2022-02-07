# Derivation of equations for thick CPWG

What we have:

* Equations for thick CPWG without dielectric
    \* Usual equations for thin CPW
    \* Correction for metal thickness
* Equations for thin CPWG with dielectric

Assumptions:

* Quasistatic approximation
* Additional capacitance due to thick metal is the same for the cases with and without dielectric
* Dielectric does not have magnetic properties so that line inductance does not change with dielectric

Basic equations:
$$ v = \frac{1}{\sqrt{LC}} $$
$$ Z = \sqrt{\frac{L}{C}} $$
$$ \Rightarrow C=\frac{1}{Zv}$$
where $v$ is phase velocity

Let $c$=speed of light

For air dielectric;
$$C=\frac{1}{cZ\_{thin,air}}$$
$$C+C\_x=\frac{1}{cZ\_{thick,air}}$$
$$\Rightarrow C\_x=\frac{1}{cZ\_{thick,air}}-\frac{1}{cZ\_{thin,air}}$$
where $C\_x$ is the extra capacitance due to the thickness of the strip.

This capacitance is the result of 3 factors. Parallel-plate capacitance between the line and side ground ($C\_p$) and fringing fields above the line ($C\_{f,above}$) and below the line ($C\_{f,below}$).

$$C\_p = 2\epsilon\_r \epsilon\_0 \frac{t\_h}{s}$$

We assume that:
$$ C\_{f,above}=C\_{f,below}=C\_f$$
$$\Rightarrow C\_x=2C\_f+C\_p$$

![](cpw_air_caps.png)


<img width="300"  src="cpw_air_caps.png" >


In the presence of dielectric, only $C\_{f,below}$ is multiplied by $\epsilon\_r$. So the additional capacitance due to thickness in the presence of dielectric is:
$$C\_{x,diel}=(1+\epsilon\_r)C\_f+C\_p$$

For CPW with dielectric and thin metal;
$$C\_{thin,diel}=\frac{1}{v\_{thin}Z\_{thin,diel}}$$
$$L\_{thin}=\frac{Z\_{thin,diel}}{v\_{thin}}$$

For CPW with dielectric and thick metal;
$$L\_{thick}=\frac{Z\_{thick,air}}{c}$$
$$v\_{thick}=\frac{1}{L\_{thick}(C\_{thin,diel}+C\_{x,diel})}$$
$$Z\_{thick,diel}=\frac{L\_{thick}}{(C\_{thin,diel}+C\_{x,diel})}$$