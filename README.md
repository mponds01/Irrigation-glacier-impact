# Karakoram

"""This file contains all the required code for running the simulation on impact of irrigation on the Karakoram glacier anomaly"""

It is build up out of the following sections

1. OGGM Tutorials (Used for acquanting myself with OGGM)
2. Climate Data Processing to define the monthly, seasonal and yearly 30-yr average perturbations 
    (Downloaded climate data from vsc Hortense (pre-processing done with cdo))
3. Glacier simulations
    3.1/2 selecting test glaciers based on Hugonnet and WGMS data availability
    3.3 coompare mass balances from modelled data compared to Hugonnet observations (20 yr avg), using pre-processing level 5
    3.4 check model performance on all glaciers in region 13 with A>10km2, using pre-processing level 5
    3.4 set up OGGM using custom climate data 