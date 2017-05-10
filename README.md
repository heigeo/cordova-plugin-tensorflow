# cordova-plugin-tensorflow

Integrate the TensorFlow inference library into your PhoneGap/Cordova application!

```javascript
var tf = new TensorFlow('inception-v1');
var imgData = "/9j/4AAQSkZJRgABAQEAYABgAAD//gBGRm ...";

tf.classify(imgData).then(function(results) {
    results.forEach(function(result) {
        console.log(result.title + " " + result.confidence);
    });
});

/* Output:
military uniform 0.647296
suit 0.0477196
academic gown 0.0232411
*/
```
## Installation

### Cordova
```bash
cordova plugin add https://github.com/heigeo/cordova-plugin-tensorflow
```

### PhoneGap Build
```xml
<!-- config.xml -->
<plugin spec="https://github.com/heigeo/cordova-plugin-tensorflow.git" />
```

## Supported Platforms

 * Android
 * iOS

## API

The plugin provides a `TensorFlow` class that can be used to initialize graphs and run the inference algorithm.

### Initialization

```javascript
// Use the Inception model (will be downloaded on first use)
var tf = new TensorFlow('inception-v1');

// Use a custom retrained model
var tf = new TensorFlow('custom-model', {
    'label': 'My Custom Model',
    'model_path': "https://example.com/graphs/custom-model-2017.zip#rounded_graph.pb",
    'label_path': "https://example.com/graphs/custom-model-2017.zip#retrained_labels.txt",
    'input_size': 299,
    'image_mean': 128,
    'image_std': 128,
    'input_name': 'Mul',
    'output_name': 'final_result'
})
```

To use a custom model, follow the steps to [retrain the model](https://www.tensorflow.org/tutorials/image_retraining) and [optimize it for mobile use](https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/).
Put the .pb and .txt files in a HTTP-accessible zip file, which will be downloaded via the [FileTransfer plugin](https://cordova.apache.org/docs/en/latest/reference/cordova-plugin-file-transfer/).  If you use the generic Inception model it will be downloaded from [the TensorFlow website](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip) on first use.

### Methods

Each method returns a `Promise` (if available) and also accepts a callback and errorCallback.


### classify(image[, callback, errorCallback])
Classifies an image with TensorFlow's inference algorithm and the registered model.  Will automatically download and initialize the model if necessary, but it is recommended to call `load()` explicitly for the best user experience.

Note that the image must be provided as base64 encoded JPEG or PNG data.  Support for file paths may be added in a future release.

```javascript
var tf = new TensorFlow(...);
var imgData = "/9j/4AAQSkZJRgABAQEAYABgAAD//gBGRm ...";
tf.classify(imgData).then(function(results) {
    results.forEach(function(result) {
        console.log(result.title + " " + result.confidence);
    });
});
```

### load()

Downloads the referenced model files and loads the graph into TensorFlow.

```javascript
var tf = new TensorFlow(...);
tf.load().then(function() {
    console.log("Model loaded");
});
```

Downloading the model files can take some time.  If you would like to provide a progress indicator, you can do that with an `onprogress` event:
```javascript
var tf = new TensorFlow(...);
tf.onprogress = function(evt) {
    if (evt['status'] == 'downloading')
        console.log("Downloading model files...");
        console.log(evt.label);
        if (evt.detail) {
            // evt.detail is from the FileTransfer API
            var $elem = $('progress');
            $elem.attr('max', evt.detail.total);
            $elem.attr('value', evt.detail.loaded);
        }
    } else if (evt['status'] == 'unzipping') {
        console.log("Extracting contents...");
    } else if (evt['status'] == 'initializing') {
        console.log("Initializing TensorFlow");
    }
};
tf.load().then(...);
```

### checkCached()
Checks whether the requisite model files have already been downloaded.  This is useful if you want to provide an interface for downloading and managing TensorFlow graphs that is separate from the classification interface.

```javascript
var tf = new TensorFlow(...);
tf.checkCached().then(function(isCached) {
    if (isCached) {
        $('button#download').hide();
    }
});
```

## References

This plugin is made possible by the following libraries and tutorials:

Source | Files
-------|--------
[TensorFlow Android Inference Interface] | [libtensorflow_inference.so],<br>[libandroid_tensorflow_inference_java.jar]
[TensorFlow Android Demo] |[Classifer.java],<br>[TensorFlowImageClassifier.java][TensorFlowImageClassifier.java] (modified)
[TensorflowPod] | Referenced via [podspec]
[TensorFlow iOS Examples] | [ios_image_load.mm][ios_image_load.mm] (modified),<br>[tensorflow_utils.mm][tensorflow_utils.mm] (+ RunModelViewController.mm)

[TensorFlow Android Inference Interface]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android
[libtensorflow_inference.so]: https://github.com/heigeo/cordova-plugin-tensorflow/blob/master/src/android/tf_libs/armeabi-v7a/libtensorflow_inference.so
[libandroid_tensorflow_inference_java.jar]: https://github.com/heigeo/cordova-plugin-tensorflow/blob/master/src/android/tf_libs/libandroid_tensorflow_inference_java.jar
[TensorFlow Android Demo]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
[Classifer.java]: https://github.com/heigeo/cordova-plugin-tensorflow/blob/master/src/android/tf_libs/Classifier.java
[TensorFlowImageClassifier.java]: https://github.com/heigeo/cordova-plugin-tensorflow/blob/master/src/android/tf_libs/TensorFlowImageClassifier.java
[TensorflowPod]: https://github.com/rainbean/TensorflowPod
[podspec]: https://github.com/heigeo/cordova-plugin-tensorflow/blob/master/plugin.xml#L38
[TensorFlow iOS Examples]: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/ios_examples
[ios_image_load.mm]: https://github.com/heigeo/cordova-plugin-tensorflow/blob/master/src/ios/tf_libs/ios_image_load.mm
[tensorflow_utils.mm]: https://github.com/heigeo/cordova-plugin-tensorflow/blob/master/src/ios/tf_libs/tensorflow_utils.mm
