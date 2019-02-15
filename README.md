This project is on "k-center clustering with outliers", as described in the papers in the papers section of the project. The two implementations of the project are the 1. robust k-center clustering offline and 2. the k-center streaming clustering without parallelisation. 

To have a quick view of the results, look at the analysis of the two algorithms using 1 million geotagged tweets in the report in the folder. 

To run the algorithm, 
for the offline version : ```python3 offline-k-center.py <no-of-clusters> <epsilon> <no-of-outliers> <no-of-tweets-to-be-clustered>```
  
  where epsilon is from 0-1 depending on the coarseness of the radius you want to find,
  and the number of points are from 1 to 1 million
  
 for the streaming version: ``` python3 k-center-streaming.py <no-of-clusters> <no-of-outliers> <alpha> <beta> <n>```
 
  where alpha = 4, beta = 8, n = 16 as values given in the paper.
  
 For any further questions, please contact amrita.suresh@ens-paris-saclay.fr
 
 Please note that for plotting the clusters, cartopy is required. In order to install cartopy, the user faced various pending bugs while doing a direct install. For best results, ```conda install -c conda-forge cartopy ``` is the smoothest way to go forward.
 
 If that is to be skipped, the clusters can still be found by omitting the plotting. Contact the author for further details.
