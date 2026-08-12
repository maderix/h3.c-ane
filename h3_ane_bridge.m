#import "h3_ane_bridge.h"

#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

struct h3_ane_model {
    void *model;
    void *request;
    char *staging_directory;
    double compile_seconds;
    bool cache_hit;
};

static void bridge_fail(char *error, size_t error_size, const char *format,
                        ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double bridge_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int h3_ane_bridge_available(void) {
    static int state = -1;
    if (state >= 0) return state;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine", RTLD_NOW);
    state = NSClassFromString(@"_ANEInMemoryModelDescriptor") != nil &&
            NSClassFromString(@"_ANEInMemoryModel") != nil &&
            NSClassFromString(@"_ANERequest") != nil &&
            NSClassFromString(@"_ANEIOSurfaceObject") != nil;
    return state;
}

IOSurfaceRef h3_ane_bridge_surface(size_t bytes) {
    size_t aligned = (bytes + 16383u) & ~(size_t)16383u;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0});
}

bool h3_ane_cache_enabled(void) {
    const char *env = getenv("H3_ANE_CACHE");
    return !env || atoi(env) != 0;
}

static void bridge_write_sources(NSString *directory, NSData *program,
                                 NSData *weights) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files createDirectoryAtPath:
        [directory stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [program writeToFile:
        [directory stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    [weights writeToFile:
        [directory stringByAppendingPathComponent:@"weights/weight.bin"]
        atomically:YES];
}

static NSString *bridge_cache_entry(NSString *identifier) {
    return [[NSTemporaryDirectory()
        stringByAppendingPathComponent:@"h3-ane-cache"]
        stringByAppendingPathComponent:identifier];
}

/* Hardlink a directory tree (same volume, so links are free); copy as the
 * fallback. The destination is replaced. */
static bool bridge_mirror(NSString *from, NSString *to) {
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:to error:nil];
    [files createDirectoryAtPath:[to stringByDeletingLastPathComponent]
        withIntermediateDirectories:YES attributes:nil error:nil];
    if ([files linkItemAtPath:from toPath:to error:nil]) return true;
    [files removeItemAtPath:to error:nil];
    return [files copyItemAtPath:from toPath:to error:nil];
}

/* The model unload deletes its staging directory, so compiled artifacts are
 * preserved in a content-addressed entry next to it and restored on reuse. */
static bool bridge_cache_restore(NSString *identifier, NSString *directory) {
    NSString *entry = bridge_cache_entry(identifier);
    NSFileManager *files = [NSFileManager defaultManager];
    if (![files fileExistsAtPath:
            [entry stringByAppendingPathComponent:@"compiled.ok"]])
        return false;
    if (!bridge_mirror(entry, directory)) return false;
    [files removeItemAtPath:
        [directory stringByAppendingPathComponent:@"compiled.ok"] error:nil];
    return true;
}

/* Only the compiled artifacts are kept: `data` embeds the constants, so the
 * weights copy and the MIL text are dead weight in an entry. */
static void bridge_cache_store(NSString *identifier, NSString *directory) {
    NSString *entry = bridge_cache_entry(identifier);
    NSFileManager *files = [NSFileManager defaultManager];
    [files removeItemAtPath:entry error:nil];
    if (![files createDirectoryAtPath:entry withIntermediateDirectories:YES
                           attributes:nil error:nil]) return;
    for (NSString *name in
         [files contentsOfDirectoryAtPath:directory error:nil]) {
        if ([name isEqualToString:@"weights"] ||
            [name isEqualToString:@"model.mil"]) continue;
        NSString *from = [directory stringByAppendingPathComponent:name];
        NSString *to = [entry stringByAppendingPathComponent:name];
        if (![files linkItemAtPath:from toPath:to error:nil]) {
            [files removeItemAtPath:to error:nil];
            if (![files copyItemAtPath:from toPath:to error:nil]) return;
        }
    }
    [[NSData data] writeToFile:
        [entry stringByAppendingPathComponent:@"compiled.ok"] atomically:YES];
}

static void bridge_cache_evict(const char *identifier) {
    if (!identifier) return;
    [[NSFileManager defaultManager]
        removeItemAtPath:bridge_cache_entry(@(identifier)) error:nil];
}

h3_ane_model *h3_ane_model_create(const char *name, const char *mil,
                                  void *weight_bytes_owned, size_t weight_bytes,
                                  IOSurfaceRef *input_surfaces,
                                  uint32_t input_count, IOSurfaceRef output,
                                  char *error, size_t error_size) {
    if (!h3_ane_bridge_available()) {
        free(weight_bytes_owned);
        bridge_fail(error, error_size, "the Neural Engine bridge is "
                    "unavailable");
        return NULL;
    }
    h3_ane_model *handle = calloc(1, sizeof(*handle));
    if (!handle) {
        free(weight_bytes_owned);
        bridge_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    @autoreleasepool {
        NSError *failure = nil;
        NSData *weights = [NSData dataWithBytesNoCopy:weight_bytes_owned
                                               length:weight_bytes
                                         freeWhenDone:YES];
        NSData *program = [NSData dataWithBytes:mil length:strlen(mil)];
        Class descriptorClass =
            NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin":
                       @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) {
            bridge_fail(error, error_size, "ANE %s descriptor rejected", name);
            free(handle);
            return NULL;
        }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) {
            bridge_fail(error, error_size, "ANE %s model rejected", name);
            free(handle);
            return NULL;
        }
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier));
        NSString *directory = [NSTemporaryDirectory()
            stringByAppendingPathComponent:identifier];
        NSFileManager *files = [NSFileManager defaultManager];
        bool cache = h3_ane_cache_enabled();
        bool cached = cache && bridge_cache_restore(identifier, directory);
        if (!cached) bridge_write_sources(directory, program, weights);
        handle->staging_directory = strdup(directory.UTF8String);
        double started = bridge_seconds();
        bool loaded = cached &&
            ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{},
                &failure);
        if (cached && !loaded) {
            failure = nil;
            bridge_write_sources(directory, program, weights);
        }
        if (!loaded) {
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(compileWithQoS:options:error:), 21, @{},
                    &failure)) {
                bridge_fail(error, error_size, "ANE %s compile failed: %s",
                            name, failure ?
                            failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                    model, @selector(loadWithQoS:options:error:), 21, @{},
                    &failure)) {
                bridge_fail(error, error_size, "ANE %s load failed: %s", name,
                            failure ?
                            failure.localizedDescription.UTF8String : "?");
                [files removeItemAtPath:directory error:nil];
                free(handle->staging_directory);
                free(handle);
                return NULL;
            }
            if (cache) bridge_cache_store(identifier, directory);
        }
        handle->cache_hit = loaded;
        handle->compile_seconds = bridge_seconds() - started;
        NSMutableArray *inputs = [NSMutableArray array];
        NSMutableArray *indices = [NSMutableArray array];
        for (uint32_t index = 0; index < input_count; index++) {
            [inputs addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:),
                input_surfaces[index])];
            [indices addObject:@(index)];
        }
        id wrapped = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:), output);
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))
                      objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            inputs, indices, @[wrapped], @[@0], nil, nil, @0);
        if (!request) {
            bridge_fail(error, error_size, "ANE %s request rejected", name);
            h3_ane_model_free(handle);
            return NULL;
        }
        handle->model = (__bridge_retained void *)model;
        handle->request = (__bridge_retained void *)request;
    }
    return handle;
}

int h3_ane_model_eval(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model || !handle->request) {
        bridge_fail(error, error_size, "the ANE model is not loaded");
        return 0;
    }
    int ok = 0;
    @autoreleasepool {
        NSError *failure = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            (__bridge id)handle->model,
            @selector(evaluateWithQoS:options:request:error:), 21, @{},
            (__bridge id)handle->request, &failure) ? 1 : 0;
        if (!ok)
            bridge_fail(error, error_size, "ANE evaluation failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
    }
    return ok;
}

int h3_ane_model_unload(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model) {
        bridge_fail(error, error_size, "no ANE model to unload");
        return 0;
    }
    @autoreleasepool {
        NSError *failure = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                (__bridge id)handle->model, @selector(unloadWithQoS:error:), 21,
                &failure)) {
            bridge_fail(error, error_size, "ANE unload failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
            return 0;
        }
    }
    return 1;
}

int h3_ane_model_reload(h3_ane_model *handle, char *error, size_t error_size) {
    if (!handle || !handle->model || !handle->staging_directory) {
        bridge_fail(error, error_size, "no ANE model to reload");
        return 0;
    }
    @autoreleasepool {
        NSString *directory = @(handle->staging_directory);
        bridge_cache_restore(directory.lastPathComponent, directory);
        NSError *failure = nil;
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                (__bridge id)handle->model,
                @selector(loadWithQoS:options:error:), 21, @{}, &failure)) {
            bridge_fail(error, error_size, "ANE reload failed: %s",
                        failure ?
                        failure.localizedDescription.UTF8String : "?");
            return 0;
        }
    }
    return 1;
}

void h3_ane_model_free(h3_ane_model *handle) {
    if (!handle) return;
    @autoreleasepool {
        if (handle->model) {
            id model = (__bridge_transfer id)handle->model;
            NSError *failure = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &failure);
        }
        if (handle->request) {
            id request = (__bridge_transfer id)handle->request;
            (void)request;
        }
        if (handle->staging_directory) {
            [[NSFileManager defaultManager]
                removeItemAtPath:@(handle->staging_directory) error:nil];
            if (!h3_ane_cache_enabled())
                bridge_cache_evict(strrchr(handle->staging_directory, '/') ?
                    strrchr(handle->staging_directory, '/') + 1 :
                    handle->staging_directory);
            free(handle->staging_directory);
        }
    }
    free(handle);
}

double h3_ane_model_compile_seconds(const h3_ane_model *handle) {
    return handle ? handle->compile_seconds : 0.0;
}

bool h3_ane_model_cache_hit(const h3_ane_model *handle) {
    return handle ? handle->cache_hit : false;
}
