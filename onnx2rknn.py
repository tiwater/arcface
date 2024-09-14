from rknn.api import RKNN

def build_rknn_model(onnx_model_path, quantize=True):
    rknn = RKNN()

    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[[127.5, 127.5, 127.5]],
                quant_img_RGB2BGR=False,
#                reorder_channel='0 1 2', # For RGB, '2 1 0' for BGR
                optimization_level=3,
                quantized_algorithm='normal', target_platform='rk3588')
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_model_path, inputs=['input'], input_size_list=[[1,3,112,112]])
    if ret != 0:
        print('Load ONNX model failed!')
        return ret
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=quantize, dataset='./dataset.txt')
    if ret != 0:
        print('Build RKNN model failed!')
        return ret
    print('done')

    print('--> Export RKNN model')
    ret = rknn.export_rknn('./model/arcface.rknn')
    if ret != 0:
        print('Export RKNN model failed!')
        return ret
    print('done')

    rknn.release()

    return 0

if __name__ == '__main__':
    model_path = './model/arcface.onnx'
    build_rknn_model(model_path)