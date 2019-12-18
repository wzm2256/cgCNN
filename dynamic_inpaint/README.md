1. The original dynamic texture is placed in *./Image*
2. Run
```python
    python Inpaint.py --image VIDEO/grass_3 --mask mask2.png
```
3. The masked video and the inpainted video will be in *./Inpainted*


**TODO:** 
1. This code dose not support Gram matrix based inpainting yet.
2. This code does not support grid searching. The template must be assigned by the user.  

