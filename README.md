# DRIL
___________________________________
**denoise_projection-script-only.py**
___________________________________
For performing the denoising of dicom projections.<br>
Takes as argument the folder of input projections (--input) and the path to save the denoised projections (--output).<br>
Also takes the path of model weights (--weight).<br>
<br>
**Sample Usage:**<br>
python denoise_projection-script-only.py --input=/media/pranjal/newdrive1/REAL-DBT-PROJECTIONS/Pranjal-PMA-DATA/04140608/LE-L-CC/ --output=/media/pranjal/newdrive1/REAL-DBT-PROJECTIONS/Pranjal-PMA-DATA/04140608/LE-L-CC-CLEAN-TEMP1/ --weight=/media/pranjal/newdrive1/DBT-PROJ-DENOISE/normal-to-three-0.99-weights/generator_weights_3550.h5
___________________________________
___________________________________
**recon-script.py**
___________________________________
For performing the reconstruction.<br>
Takes as argument the folder of input projections (--input) and the path to save the denoised projections (--output).<br>
orientation = left or right<br> 
neighbours  = Number of neighbours on one side to consider while calculating the prior.<br>
prior       = huber or quadratic or anisotropic_quadratic<br>
Other arguments are self explnatory.<br>
<br>
**Sample Usage:**<br>
python recon-script.py --input=/media/drilnvm/ubuntudata2/REAL-DBT-PROJECTIONS/MASS/MC-20/LE-R-CC/ --output=/media/drilnvm/ubuntudata2/IMPI_recons/ --orientation=left --prior=anisotropic_quadratic --name=test2 --beta=0.5,0.8 --delta=0.0006 --lambdavalue=0.9 --neighbours=1
