Ideas generales:
Determinar los pesos que minimizan MSE, usando pair-wise testing.
Framework que trabaja con el espacio-imagen para obtener low-error estimates for any *quantity of interest*. *Preguntar*
Otras aplicaciones de este approach para RR y MIS (Multiple Importance Sampling).


Datos que calcular online:
- G-Buffer(albedo, normales, etc.) | Para el joint bilateral filter.
- Estadísticas descriptivas de la distribución de los samples (mean, variance, skewness, kurtosis, etc.) | Para el pair-wise testing.

Intervalos de confianza (IMPORTANTE)
CLAVE: Uso de test-estadísticos relajados mediante el uso de high-order central moments, ya que normalmente se asume una distribución normal.
 
$\hat{\theta}$ (Estimador de radianza de cada píxel, unbias y noisy)
$\tilde{\theta}_j = \sum_{i} w_{ij} \hat{\theta}_i$ (Denoised estimator, bias)

CLAVE: Encontrar un peso optimo para reducir el MSE.
$MSE(\tilde{\theta}_j, \theta_j) = E\left[(\tilde{\theta}_j - \theta_j)^2\right] = \text{Var}(\tilde{\theta}_j) + \text{Bias}(\tilde{\theta}_j, \theta_j)^2$
En un mundo perfecto donde tenemos los valores reales sería muy fácil calcular los pesos que minimizan el MSE.
CLAVE: Se formula una memebership function, en la que se siguen unas propiedades que hacen más probable que la combinación mejore la imagen.
Framework: $w_{ij} = \frac{\rho_{ij} m_{ij}}{\sum_{i} \rho_{ij} m_{ij}}$
$\rho_{ij}$: Joint Bilateral Filter, muy efectivo para reducir varianza
$m_{ij}$: Funcion de membership, límita el bias provocado por el filtro excluyendo estiamdores

El problema de minimización se formula para encontrar $w^*$, el peso óptimo en la combinación de los estimadores 
$\hat{\theta_i}$ $\hat{\theta_j}$  , minimizando el MSE.
Dado que los estmidaores $\hat{\theta_i}$ $\hat{\theta_j}$ son ruidosos los pesos $w^*$ también tienen ruido.
Por lo que se trabaja con un $\rho_{ij}$ preexistente y una función de membership binaría. 
La función de membership binaria decide si dos estimadores pueden combinarse basándose en un test estadístico con un umbral de confianza.
El test estádistico usa $w^*$, que es el peso optimo para los estimadores $\hat{\theta_i}$ $\hat{\theta_j}$ y $\theta_i$ $\theta_i$
*Preguntar* No entiendo muy bien que minimizan los pesos $w^*$ y no tienes $\theta_i$ $\theta_i$

IMPORTANTE: En lugar de usar directamente la radiancia de cada muestra, se aplica una transformación Box-Cox para normalizar los valores antes de calcular las estadísticas.

Conclusiones:
El denoiser se forma por unos pesos a priori $\rho_{ij}$ que se calculan usando un filtro a priorio como 
joint bilateral filter, se usa una funcion de membership $m_{ij}$ binaria para decidir si los estimadores mejoran la imagen.
Para calcular $m_{ij}$ se usan test estadísticos que usan los pesos $w^*$ que se obtendrían al mimizar el MSE de usar ambos estimadores.
Usar box-cox y correcting  the mean [Curto 2023] para que el método sea más robusto.




