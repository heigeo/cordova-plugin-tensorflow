#include <memory>
#include <vector>
#include "tensorflow/core/public/session.h"

#import "ios_image_load.h"
#import "tensorflow_utils.h"
#import <Cordova/CDVPlugin.h>

@interface TensorFlow : CDVPlugin {
    NSMutableDictionary *classifiers;
}

- (void)loadModel:(CDVInvokedUrlCommand*)command;
- (void)classify:(CDVInvokedUrlCommand*)command;

@end

@interface Classifier : NSObject {
    std::unique_ptr<tensorflow::Session> session;
    NSString* model_file;
    NSString* label_file;
    std::vector<std::string> labels;
    int input_size;
    int image_mean;
    float image_std;
    NSString* input_name;
    NSString* output_name;
}

 - (id)initWithModel:(NSString *)model_file_
       label_file:(NSString *)label_file_
       input_size:(int)input_size_
       image_mean:(int)image_mean_
       image_std:(float)image_std_
       input_name:(NSString *)input_name_
       output_name:(NSString *)output_name_;
- (tensorflow::Status)load;
- (tensorflow::Status)classify:(NSString *)image results:(NSMutableArray *)results;

@end
