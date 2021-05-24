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

**recon-script.py**
___________________________________
For performing the reconstruction.<br>
Takes as argument the folder of input projections (--input) and the path to save the denoised projections (--output).<br>
Other arguments are self explnatory.<br>
<br>
**Sample Usage:**<br>
python recon-script.py --input=/media/drilnvm/ubuntudata2/REAL-DBT-PROJECTIONS/MASS/MC-20/LE-R-CC/ --output=/media/drilnvm/ubuntudata2/IMPI_recons/ --orientation=right --prior=huber --name=test1 --beta=0.15,0.251,0.35 --delta=0.0005 --lambdavalue=0.9
