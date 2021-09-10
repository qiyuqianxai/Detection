from mmdet.apis import init_detector, inference_detector


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.1,
                       title='result',
                       wait_time=0,
                       out_file=None):
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        out_file=out_file)

config_file = 'configs/yolo/yolov3_outdoor_coco.py'

checkpoint_file = 'tools/work_dirs/yolov3_outdoor_coco/epoch_84.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img_name = '/BOBO/datasets/detection_data/val2017/LZ63.jpg'
result = inference_detector(model, img_name)
show_result_pyplot(model,img_name,result,out_file="predict_res.jpg")