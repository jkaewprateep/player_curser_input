# player_curser_input

Study games player cursor as image input, most of the games that we are required to know player position on the screen and objects to perform efficiency action continue without game's cursers we can do it by these methods as samples:

### 1. Image enchantments : ### 

Extracting features or masking, significant of data can decided by its types, data contrast, data dimensions, input widths, input layers and criticals.

```
input = tf.constant( input, dtype=tf.int32 ).numpy()
inputs[26:35, 17:26, 0:1] = 255.0
masked_input = tf.keras.layers.Masking( mask_value=255.0, input_shape=(timesteps, features) )( inputs )
```

### 2. Horizontals : ###

When input image contrast is the neighbours contrast or image global contrast, the horizontals is compared within the same layer. Continue, the layers can be channels or the same layer it can be extracting of some relative data S1 = { 0.89, 0.89, 0.56, 0.73, 0.35, 0.45 ... } 👧💬 Can we noticed some significants value from this series S1 ⁉️ 👧💬 Try ti use the convolutions function to the inputs data ```tf.keras.layers.Conv1D( 1, 3, activation='relu')(S1)``` you have ```{ 0.78, 0.73, 0.55, 0.51 ... }``` OR { 🟩, 🟦 } from [tf.image.resize()](https://www.tensorflow.org/api_docs/python/tf/image/resize)

### 3. Data grids and Segmentation :  ###

#### 3.1 crop_to_bounding_box ####

We can select the interesting scope from the object radious, in games or the game player curser as inputs[26:35, 17:26, 0:1] or you can create specific shapes for inputs but the rectangular is easiest matching with the same line environments or they can use hexagornal rectangular or eclipse or the elapse pictures but our requirements is extracting objects's positions from the input image as its input to our neuron networks [tf.image.crop_to_bounding_box()](https://www.tensorflow.org/api_docs/python/tf/image/crop_to_bounding_box)

```
offset_height = 26
offset_width = 17
target_height = 35 - 26
target_width = 26 - 17
cropped_original_image = tf.image.crop_to_bounding_box( original_image, offset_height, offset_width, target_height, target_width )
```

#### 3.2 draw_bounding_boxes #### 

We can create significants and display our games curser by draws them and remarks their value as the same as in the Pixel Helicopter games and other games, x and y positioning as the side product of ```[26/42, 17/42, 35/42, 26/42]``` [tf.image.draw_bounding_boxes()](https://www.tensorflow.org/api_docs/python/tf/image/draw_bounding_boxes)

```
boxes = tf.constant([26/42, 17/42, 35/42, 26/42], shape=(1, 1, 4))
colors = tf.constant([[144.0, 238.0, 144.0]])
original_image = tf.image.draw_bounding_boxes( tf.expand_dims( original_image, axis=0 ), boxes, colors)
```

#### 3.3 image.resize ####

We consider the ```nearest```, ```bilinear```, ```area```, ```gaussian``` and ```lanczos5``` in the ```method``` parameter, they are working by fiters inputs and provide its output from their relative values inside the selected matrixes. [tf.image.resizes()](https://www.tensorflow.org/api_docs/python/tf/image/resize)

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



### 4. Masking : ###

Resizing and Rescaling, they are working on the data distributions and the data relative with data masking, masked data will not tobe process and leaves with the original information. [tf.keras.layers.Masking()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking)
	
```
image = tf.image.resize(observation, [32, 32])
image = tf.image.rgb_to_grayscale( tf.cast( tf.keras.utils.img_to_array( image ), dtype=tf.float32 ) )

input = tf.constant( input, dtype=tf.int32 ).numpy()
inputs[26:35, 17:26, 0:1] = 255.0
masked_input = tf.keras.layers.Masking( mask_value=255.0, input_shape=(timesteps, features) )( inputs )

result_image = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(masked_input)
```

### Random Functions ###

How do we create a data relationship, quivalent or equation for our networks model learning and it can apply to the function responses in the game environments. From actions spaces, action created from 3 values as floating numbers in an array representing from ```1.0 to -1.0``` as ```turn wheel```, ```engine accleration```, and ```break```. The sample action value are ```[ 0.5, 0.5, 0.0 ]```, ```[ 0.0, 0.6, 0.0]```, ```[ -0.5, 0.5, 0.0 ]```, and ```[ 0.0, 0.0, 0.5 ]``` for ```turn-right```, ```engine accelerate```, ```turn-left```, and ```breaks```  

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

### Input image as curser ###

It is required to transfroming the scoped image into matrixes value for the networks model input, the model catagorized object and determine the response functions.

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

### Model ###

The model response as an array of 3 values mapping to the action spaces, ```action = [ 0.0, 0.5, 0.0 ]``` and it isour program requirements tell the game of our selected action to player in the game environments ```turn-right```, ```engine accelerate```, ```turn-left```, and ```breaks``` to ```observation, reward, done, info, prob = env.step(action)```

```
model = tf.keras.models.Sequential([
	tf.keras.layers.InputLayer(input_shape=INPUT_DIMS),

	tf.keras.layers.Normalization(mean=3., variance=2.),
	tf.keras.layers.Normalization(mean=4., variance=6.),
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Reshape((128, 9)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96, return_sequences=True, return_state=False)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(96)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(192, activation='relu'),
	tf.keras.layers.Dense(3),
])

model.summary()
```

### Files and Directory ###

| File name     | Description   |
| ------------- |:-------------:|
| Car Racing.gif 	| result from program |
| Figure_1.png      	| curser and bounding box      | 
| 02.png 	| Pixel Helicopter games      | 
| README.md 	| readme file      | 

## Result image ##

There are some results from our simple codes implementation.

#### Play game ####

The car racing game play with Tensorflow.

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/Car%20Racing.gif?raw=true "Title")

#### Remark curser display ####

The game's player bounding box matching for curser positions.

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/Figure_1.png?raw=true "Title")

#### Application with other games #### 

The Pixel Helicopter game.

![Alt text](https://github.com/jkaewprateep/player_curser_input/blob/main/02.png?raw=true "Title")
