1. The exemplar image texture is placed in *./Image*, and the trained model is placed in *./Saved_model*
2. Make sure you can synthesize textures using the pretrained model.
```
python Test.py peppers.jpg --step 2200 --size 1024
```
3. The synthesized image texture will be in *./Produce_Generator*.
    
    **Notice:** the output image will be 128 pixels smaller than the given size parameter,    as we crop the border pixels for better quality.
4. Train your model using your data.
```
python Train.py python Train.py peppers.jpg
```
5. The trained model will be in *./Saved_model*
