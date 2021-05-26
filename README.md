# DRIL
___________________________________
**denoise_projection-script-only.py**
___________________________________
For performing the denoising of dicom projections.<br>
Arguments to the script<br>
**--input**=Path of input projections.<br>
**--output**=Path to save the denoised projections.<br>
**--weight**=Path of model weights.<br>
<br>
**Sample Usage:**<br>
python denoise_projection-script-only.py --input=/media/pranjal/newdrive1/REAL-DBT-PROJECTIONS/Pranjal-PMA-DATA/04140608/LE-L-CC/ --output=/media/pranjal/newdrive1/REAL-DBT-PROJECTIONS/Pranjal-PMA-DATA/04140608/LE-L-CC-CLEAN-TEMP1/ --weight=/media/pranjal/newdrive1/DBT-PROJ-DENOISE/normal-to-three-0.99-weights/generator_weights_3550.h5
___________________________________
___________________________________
**recon-script.py**
___________________________________
For performing reconstruction of the projections.<br>
Arguments to the script<br>
**--input**=Path of input projections.<br>
**--output**=Path to save the reconstructed volume.<br>
**--orientation**=left or right<br> 
**--neighbours**=Number of neighbours on one side to consider while calculating the prior.<br>
**--prior**=huber or quadratic or anisotropic_quadratic<br>
**--name**=Name of the volume for storing<br><br>
Other arugments are **--beta**, **--delta**, **--lambdavalue**.<br>
For **beta** and **delta** please enter comma separated values if reconstruction for multiple values needs to be done.<br>
For changing the geometry please edit the script as we do in previous reconstruction code. It will require calculation using the excel script.
<br><br>
**Sample Usage:**<br>
python recon-script.py --input=/media/drilnvm/ubuntudata2/REAL-DBT-PROJECTIONS/MASS/MC-20/LE-R-CC/ --output=/media/drilnvm/ubuntudata2/IMPI_recons/ --orientation=left --prior=anisotropic_quadratic --name=test2 --beta=0.5,0.8 --delta=0.0006 --lambdavalue=0.9 --neighbours=1
