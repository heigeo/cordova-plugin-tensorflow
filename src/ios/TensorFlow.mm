#import "TensorFlow.h"
#import <Cordova/CDVPlugin.h>
#import "ios_image_load.h"
#import "tensorflow_utils.h"

@implementation TensorFlow

- (void)loadModel:(CDVInvokedUrlCommand*)command
{
    CDVPluginResult* pluginResult = nil;
    NSString* model_name = [command.arguments objectAtIndex:0];
    Classifier* classifier = [
        [Classifier alloc]
        initWithModel:[command.arguments objectAtIndex:1]
        label_file:[command.arguments objectAtIndex:2]
        input_size:[(NSNumber *)[command.arguments objectAtIndex:3] intValue]
        image_mean:[(NSNumber *)[command.arguments objectAtIndex:4] intValue]
        image_std:[(NSNumber *)[command.arguments objectAtIndex:5] floatValue]
        input_name:[command.arguments objectAtIndex:6]
        output_name:[command.arguments objectAtIndex:7]
    ];
    if (classifiers == nil) {
        classifiers = [NSMutableDictionary dictionaryWithDictionary:@{}];
    }
    classifiers[model_name] = classifier;
    tensorflow::Status result = [classifier load];
    if (result.ok()) {
        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK];
    } else {
        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_ERROR];
    }

    [self.commandDelegate sendPluginResult:pluginResult callbackId:command.callbackId];
}

- (void)classify:(CDVInvokedUrlCommand*)command
{
    CDVPluginResult* pluginResult = nil;
    NSString* model_name = [command.arguments objectAtIndex:0];
    NSString* image = [command.arguments objectAtIndex:1];
    Classifier* classifier = classifiers[model_name];
    NSMutableArray* results = [[NSMutableArray alloc] init];
    tensorflow::Status result = [classifier classify:image results:results];
    if (result.ok()) {
        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsArray:results];
    } else {
        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_ERROR];
    }

    [self.commandDelegate sendPluginResult:pluginResult callbackId:command.callbackId];
}

@end

@implementation Classifier

 - (id)initWithModel:(NSString *)model_file_
       label_file:(NSString *)label_file_
       input_size:(int)input_size_
       image_mean:(int)image_mean_
       image_std:(float)image_std_
       input_name:(NSString *)input_name_
       output_name:(NSString *)output_name_
{
    self = [super init];
    if (self) {
        model_file = model_file_;
        label_file = label_file_;
        input_size = input_size_;
        image_mean = image_mean_;
        image_std = image_std_;
        input_name = input_name_;
        output_name = output_name_;
    }
    return self;
}

 - (tensorflow::Status)load
{
    tensorflow::Status result;
    result = LoadModel(model_file, &session);
    if (result.ok()) {
        result = LoadLabels(label_file, &labels);
    }
    return result;
}

 - (tensorflow::Status)classify:(NSString *)image results:(NSMutableArray *)results
{
    std::vector<Result> tfresults;
    tensorflow::Status result = RunInferenceOnImage(
        image,
        input_size,
        image_mean,
        image_std,
        [input_name UTF8String],
        [output_name UTF8String],
        &session,
        &labels,
        &tfresults
    );
    if (!result.ok()) {
        return result;
    }
    for (struct Result result : tfresults) {
        [results addObject: @{
            @"title": result.label,
            @"confidence": result.confidence
        }];
    }
}

@end
