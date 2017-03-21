package io.wq.tensorflow;

import org.apache.cordova.CordovaPlugin;
import org.apache.cordova.CallbackContext;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONException;

import android.util.Base64;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;

import org.tensorflow.demo.TensorFlowImageClassifier;
import org.tensorflow.demo.Classifier.Recognition;
import org.tensorflow.demo.Classifier;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import android.media.ThumbnailUtils;

public class TensorFlow extends CordovaPlugin {

@Override
public boolean execute(String action, JSONArray args, CallbackContext callbackContext) throws JSONException {
    if (action.equals("loadModel")) {
        this.loadModel(
            args.getString(0),
            args.getString(1),
            args.getString(2),
            args.getInt(3),
            args.getInt(4),
            (float) args.getDouble(5),
            args.getString(6),
            args.getString(7),
            callbackContext
        );
        return true;
    } else if (action.equals("classify")) {
        this.classify(args.getString(0), args.getString(1), callbackContext);
        return true;
    } else {
        return false;
    }
}


private Map<String,Classifier> classifiers = new HashMap();
private Map<String,Integer> sizes = new HashMap();

private void loadModel(String modelName, String modelFile, String labelFile,
                        int inputSize, int imageMean, float imageStd,
                        String inputName, String outputName,
                        CallbackContext callbackContext) {
    classifiers.put(modelName, TensorFlowImageClassifier.create(
        cordova.getActivity().getAssets(),
        modelFile,
        labelFile,
        inputSize,
        imageMean,
        imageStd,
        inputName,
        outputName
    ));
    sizes.put(modelName, inputSize);
    callbackContext.success();
}

private void classify(String modelName, String image, CallbackContext callbackContext) {
    byte[] imageData = Base64.decode(image, Base64.DEFAULT);
    Classifier classifier = classifiers.get(modelName);
    int size = sizes.get(modelName);
    Bitmap bitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.length);
    Bitmap cropped = ThumbnailUtils.extractThumbnail(bitmap, size, size);
    List<Recognition> results = classifier.recognizeImage(cropped);
    JSONArray output = new JSONArray();
    try {
        for (Recognition result : results) {
            JSONObject record = new JSONObject();
            record.put("title", result.getTitle());
            record.put("confidence", result.getConfidence());
            output.put(record);
        }
    } catch (JSONException e) {
    }
    callbackContext.success(output);
}


}
