function TensorFlow(modelId, model) {
    this.modelId = modelId;
    if (model) {
        model = registerModel(modelId, model);
    } else {
        model = getModel(modelId);
    }
    this.model = model;
    this.onprogress = function() {};
}


TensorFlow.prototype.load = function(successCallback, errorCallback) {
    var promise;
    if (window.Promise && !successCallback) {
        promise = new Promise(function(resolve, reject) {
            successCallback = resolve;
            errorCallback = reject;
        });
    }
    loadModel(
        this.modelId,
        successCallback,
        errorCallback,
        this.onprogress
    );
    return promise;
};

TensorFlow.prototype.checkCached = function(successCallback, errorCallback) {
    var promise;
    if (window.Promise && !successCallback) {
        promise = new Promise(function(resolve, reject) {
            successCallback = resolve;
            errorCallback = reject;
        });
    }
    checkCached(this.modelId, successCallback, errorCallback);
    return promise;
};

TensorFlow.prototype.classify = function(image, successCallback, errorCallback) {
    var promise;
    if (window.Promise && !successCallback) {
        promise = new Promise(function(resolve, reject) {
            successCallback = resolve;
            errorCallback = reject;
        });
    }

    var self = this;
    if (!self.model.loaded) {
        self.load(function() {
            if (!self.model.loaded) {
                errorCallback("Error loading model!");
                return;
            }
            self.classify(image, successCallback, errorCallback);
        }, errorCallback);
        return promise;
    }

    cordova.exec(
        successCallback, errorCallback,
        "TensorFlow", "classify", [self.modelId, image]
    );
    return promise;
};

// Internal API for downloading and caching model files
var models = {};

var FIELDS = [
    'label',

    'model_path',
    'label_path',

    'input_size',
    'image_mean',
    'image_std',
    'input_name',
    'output_name'
];

function registerModel(modelId, model) {
    FIELDS.forEach(function(field) {
        if (!model[field]) {
            throw 'Missing "' + field + '" on model description';
        }
    });

    if (model.model_path.match(/^http/) || model.label_path.match(/^http/)) {
        model.cached = false;
    } else {
        model.cached = true;
    }
    models[modelId] = model;
    model.id = modelId;
    return model;
}

function getModel(modelId) {
    var model = models[modelId];
    if (!model) {
        throw "Unknown model " + modelId;
    }
    return model;
}

var INCEPTION = 'https://storage.googleapis.com/download.tensorflow.org/models/';
registerModel('inception-v1', {
    'label': 'Inception v1',
    'model_path': INCEPTION + 'inception5h.zip#tensorflow_inception_graph.pb',
    'label_path': INCEPTION + 'inception5h.zip#imagenet_comp_graph_label_strings.txt',
    'input_size': 224,
    'image_mean': 117,
    'image_std': 1,
    'input_name': 'input',
    'output_name': 'output'
});

registerModel('inception-v3', {
    'label': 'Inception v3',
    'model_path': INCEPTION + 'inception_dec_2015.zip#tensorflow_inception_graph.pb',
    'label_path': INCEPTION + 'inception_dec_2015.zip#imagenet_comp_graph_label_strings.txt',
    'input_size': 299,
    'image_mean': 128,
    'image_std': 128,
    'input_name': 'Mul',
    'output_name': 'final_result'
});

function loadModel(modelId, callback, errorCallback, progressCallback) {
    var model;
    try {
        model = getModel(modelId);
    } catch (e) {
        errorCallback(e);
        return;
    }
    if (!progressCallback) {
        progressCallback = function(stat) {
            console.log(stat.label);
        };
    }
    if (!model.cached) {
        checkCached(modelId, function(cached) {
            if (!cached) {
                fetchModel(
                    model,
                    initClassifier,
                    errorCallback,
                    progressCallback
                );
            } else {
                initClassifier();
            }
        }, errorCallback);
    } else {
        initClassifier();
    }
    function initClassifier() {
        var modelPath = (model.local_model_path || model.model_path),
            labelPath = (model.local_label_path || model.label_path);
        modelPath = modelPath.replace(/^file:\/\//, '');
        labelPath = labelPath.replace(/^file:\/\//, '');
        progressCallback({
            'status': 'initializing',
            'label': 'Initializing classifier'
        });
        cordova.exec(function() {
            model.loaded = true;
            callback(model);
        }, errorCallback, "TensorFlow", "loadModel", [
            model.id,
            modelPath,
            labelPath,
            model.input_size,
            model.image_mean,
            model.image_std,
            model.input_name,
            model.output_name
        ]);
    }
}

function getPath(filename) {
    return (
        cordova.file.externalDataDirectory || cordova.file.dataDirectory
    ) + filename;
}

function fetchModel(model, callback, errorCallback, progressCallback) {
    fetchZip(model, callback, errorCallback, progressCallback);
}

function checkCached(modelId, callback, errorCallback) {
    var model;
    try {
        model = getModel(modelId);
    } catch (e) {
        errorCallback(e);
        return;
    }
    var zipUrl = model.model_path.split('#')[0];
    if (model.label_path.indexOf(zipUrl) == -1) {
        errorCallback('Model and labels must be in same zip file!');
        return;
    }
    var modelZipName = model.model_path.replace(zipUrl + '#', '');
    var labelZipName = model.label_path.replace(zipUrl + '#', '');
    var zipPath = getPath(model.id + '.zip');
    var dir = getPath(model.id);

    model.local_model_path = dir + '/' + modelZipName;
    model.local_label_path = dir + '/' + labelZipName;

    resolveLocalFileSystemURL(
        model.local_model_path, cached(true), cached(false)
    );

    function cached(result) {
        return function() {
            model.cached = result;
            callback(model.cached);
        };
    }
}

function fetchZip(model, callback, errorCallback, progressCallback) {
    var zipUrl = model.model_path.split('#')[0];
    var zipPath = getPath(model.id + '.zip');
    var dir = getPath(model.id);
    var fileTransfer = new FileTransfer();
    progressCallback({
        'status': 'downloading',
        'label': 'Downloading model files',
    });
    fileTransfer.onprogress = function(evt) {
        var label = 'Downloading';
        if (evt.lengthComputable) {
            label += ' (' + evt.loaded + '/' + evt.total + ')';
        } else {
            label += '...';
        }
        progressCallback({
            'status': 'downloading',
            'label': label,
            'detail': evt
        });
    };
    fileTransfer.download(zipUrl, zipPath, function(entry) {
        progressCallback({
            'status': 'unzipping',
            'label': 'Extracting contents'
        });
        zip.unzip(zipPath, dir, function(result) {
            if (result == -1) {
                errorCallback('Error unzipping file');
                return;
            }
            model.cached = true;
            callback();
        });
    }, errorCallback);
}

TensorFlow._models = models;
module.exports = TensorFlow;
