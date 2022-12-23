# player_curser_input
Study games player cursor as image input

```
original_image = tf.image.resize(observation, [42, 42])
image = tf.image.rgb_to_grayscale( tf.cast( tf.keras.utils.img_to_array( original_image ), dtype=tf.float32 ) )
image = tf.expand_dims( image, axis=0 )
	
image = tf.keras.layers.Normalization(mean=3., variance=2.)(image)
image = tf.keras.layers.Normalization(mean=4., variance=6.)(image)
image = tf.squeeze( image )
	
original_image = tf.constant( original_image ).numpy()[:,0:,:]

offset_height = 26
offset_width = 17
target_height = 35 - 26
target_width = 26 - 17
cropped_original_image = tf.image.crop_to_bounding_box( original_image, offset_height, offset_width, target_height, target_width )
```

## Result image ##

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/Car%20Racing.gif?raw=true "Title")

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/Figure_1.png?raw=true "Title")

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/02.png?raw=true "Title")
