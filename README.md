# player_curser_input
Study games player cursor as image input, most of the games we reqire to know player position in the screen and objects to perform efficiency action continue without curse we can do by these methods as sample:

1. Image enchantments : extracting features or masking, significant of data can by types, contrast, dimensions, widths, layers and criticals.

```
input = tf.constant( input, dtype=tf.int32 ).numpy()
inputs[26:35, 17:26, 0:1] = 255.0
masked_input = tf.keras.layers.Masking( mask_value=255.0, input_shape=(timesteps, features) )( inputs )
```

2. Horizontals : when contrast is neighbours contrast or image global contrast the horizontals is level compared with in same layer. In example layers can be channels or the same layer it can be some relative data when all have relative values as S1 = { 0.89, 0.89, 0.56, 0.73, 0.35, 0.45 ... } 👧💬 Can we see some significants value from this series S1 ⁉️ 👧💬 Try convolutions them ```tf.keras.layers.Conv1D( 1, 3, activation='relu')(S1)``` you have ```{ 0.78, 0.73, 0.55, 0.51 ... }``` OR { 🟩, 🟦 } from [tf.image.resize()](https://www.tensorflow.org/api_docs/python/tf/image/resize)

3. Data grids and Segmentation : 

```
offset_height = 26
offset_width = 17
target_height = 35 - 26
target_width = 26 - 17
cropped_original_image = tf.image.crop_to_bounding_box( original_image, offset_height, offset_width, target_height, target_width )
```

```
boxes = tf.constant([26/42, 17/42, 35/42, 26/42], shape=(1, 1, 4))
colors = tf.constant([[144.0, 238.0, 144.0]])
original_image = tf.image.draw_bounding_boxes( tf.expand_dims( original_image, axis=0 ), boxes, colors)
```

Consider ```nearest```, ```bilinear```, ```area```, ```gaussian``` and ```lanczos5``` in the ```method``` parameter, they are working by fiters inputs and provide output from the relative values inside the selected matrixes.

```
# Image resize method in Tensorflow
tf.image.resize(
    images,
    size,
    method=ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)
```

Resize and Rescaling, they are working on data distributions and data relative with masking the masked data will not process and leaves with original information. 
	
```
image = tf.image.resize(observation, [32, 32])
image = tf.image.rgb_to_grayscale( tf.cast( tf.keras.utils.img_to_array( image ), dtype=tf.float32 ) )

input = tf.constant( input, dtype=tf.int32 ).numpy()
inputs[26:35, 17:26, 0:1] = 255.0
masked_input = tf.keras.layers.Masking( mask_value=255.0, input_shape=(timesteps, features) )( inputs )

result_image = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(masked_input)
```

## Random Functions ##

```
# action = [ a, b, c ]
# a wheel
# b engine
# c breaks

wheels = 0.0
speed = 0.6
breaks = 0.0
	
left_side = int(tf.reduce_mean( image[1:3,:,:] ).numpy())
right_side = int(tf.reduce_mean( image[7:9,:,:] ).numpy())
	
coeff_01 = left_side
coeff_02 = tf.constant( [ left_side, right_side ] ).numpy()[ tf.math.argmin([ left_side, right_side ]).numpy() ] + 7
coeff_03 = right_side
	
wheels = tf.constant( [ coeff_01, coeff_02, coeff_03 ], shape=( 1, 3 ) )
wheels = tf.cast( wheels, dtype=tf.int32 )

action = [ wheels, speed, breaks ]
```

## Input image as curser ##

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
cropped_original_image = tf.image.crop_to_bounding_box( original_image, offset_height, offset_width, 
				target_height, target_width )
```

## Result image ##

#### Play game ####

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/Car%20Racing.gif?raw=true "Title")

#### Remark curser display ####

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/Figure_1.png?raw=true "Title")

#### Application with other games #### 

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/02.png?raw=true "Title")
