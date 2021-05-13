# DRIL

**denoise_projection-script-only.py**
___________________________________
For performing the denoising of dicom projections.<br>
Takes as arugment the folder of input projections (--input) and the path to save the denoised projections (--output).<br>
Also takes the path of model weights (--weight).<br>
<br>
**Sample Usage:**<br>
python denoise_projection-script-only.py --input=/media/pranjal/newdrive1/REAL-DBT-PROJECTIONS/Pranjal-PMA-DATA/04140608/LE-L-CC/ --output=/media/pranjal/newdrive1/REAL-DBT-PROJECTIONS/Pranjal-PMA-DATA/04140608/LE-L-CC-CLEAN-TEMP1/ --weight=/media/pranjal/newdrive1/DBT-PROJ-DENOISE/normal-to-three-0.99-weights/generator_weights_3550.h5
